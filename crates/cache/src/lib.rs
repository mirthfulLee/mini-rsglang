#![forbid(unsafe_code)]

use rsglang_core::{Error, PageId, Result};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct CacheMatch {
    pub node: usize,
    pub pages: Vec<PageId>,
}
pub struct InsertResult {
    pub handle: CacheMatch,
    pub retained: Vec<PageId>,
}
pub trait PrefixCache: Send {
    fn lookup(&mut self, tokens: &[u32]) -> CacheMatch;
    fn insert(&mut self, tokens: &[u32], pages: &[PageId]) -> InsertResult;
    fn pin(&mut self, node: usize);
    fn unpin(&mut self, node: usize);
    fn evictable_pages(&self) -> usize;
    fn evict(&mut self, count: usize) -> Vec<PageId>;
    fn resident_pages(&self) -> Vec<PageId>;
}
#[derive(Default)]
pub struct NaiveCache;
impl PrefixCache for NaiveCache {
    fn lookup(&mut self, _: &[u32]) -> CacheMatch {
        CacheMatch::default()
    }
    fn insert(&mut self, _: &[u32], _: &[PageId]) -> InsertResult {
        InsertResult {
            handle: CacheMatch::default(),
            retained: vec![],
        }
    }
    fn pin(&mut self, _: usize) {}
    fn unpin(&mut self, _: usize) {}
    fn evictable_pages(&self) -> usize {
        0
    }
    fn evict(&mut self, _: usize) -> Vec<PageId> {
        vec![]
    }
    fn resident_pages(&self) -> Vec<PageId> {
        vec![]
    }
}

#[derive(Default)]
struct Node {
    key: Vec<u32>,
    pages: Vec<PageId>,
    parent: usize,
    children: BTreeMap<Vec<u32>, usize>,
    pins: usize,
    tick: u64,
}
/// Compressed radix edges are page-aligned. Deleted arena slots are reused.
pub struct RadixCache {
    nodes: Vec<Option<Node>>,
    vacant: Vec<usize>,
    page_size: usize,
    clock: u64,
}
impl RadixCache {
    pub fn new(page_size: usize) -> Self {
        assert!(page_size > 0);
        Self {
            nodes: vec![Some(Node::default())],
            vacant: vec![],
            page_size,
            clock: 0,
        }
    }
    fn node(&self, id: usize) -> &Node {
        self.nodes[id].as_ref().expect("live radix node")
    }
    fn node_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id].as_mut().expect("live radix node")
    }
    fn add(&mut self, node: Node) -> usize {
        if let Some(id) = self.vacant.pop() {
            self.nodes[id] = Some(node);
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(Some(node));
            id
        }
    }
    fn split(&mut self, id: usize, tokens: usize) -> usize {
        let page_size = self.page_size;
        let n = self.node_mut(id);
        let parent = n.parent;
        let tail_key = n.key.split_off(tokens);
        let tail_pages = n.pages.split_off(tokens / page_size);
        let head_key = std::mem::replace(&mut n.key, tail_key);
        let head_pages = std::mem::replace(&mut n.pages, tail_pages);
        let pins = n.pins;
        let tick = n.tick;
        let selector = head_key[..page_size].to_vec();
        let child_selector = self.node(id).key[..page_size].to_vec();
        let mid = self.add(Node {
            key: head_key,
            pages: head_pages,
            parent,
            children: BTreeMap::from([(child_selector, id)]),
            pins,
            tick,
        });
        self.node_mut(parent).children.insert(selector, mid);
        self.node_mut(id).parent = mid;
        mid
    }
    fn walk(&mut self, tokens: &[u32]) -> (usize, usize) {
        self.clock += 1;
        let tick = self.clock;
        let mut id = 0;
        let mut offset = 0;
        while tokens.len() >= offset + self.page_size {
            let selector = &tokens[offset..offset + self.page_size];
            let Some(&child) = self.node(id).children.get(selector) else {
                break;
            };
            let key = &self.node(child).key;
            let common = key
                .iter()
                .zip(&tokens[offset..])
                .take_while(|(a, b)| a == b)
                .count();
            let common = common / self.page_size * self.page_size;
            if common < key.len() {
                id = self.split(child, common);
                self.node_mut(id).tick = tick;
                offset += common;
                break;
            }
            offset += common;
            id = child;
            self.node_mut(id).tick = tick;
        }
        (id, offset)
    }
    fn handle(&self, node: usize) -> CacheMatch {
        let mut path = vec![];
        let mut id = node;
        while id != 0 {
            path.push(id);
            id = self.node(id).parent;
        }
        let pages = path
            .into_iter()
            .rev()
            .flat_map(|n| self.node(n).pages.iter().copied())
            .collect();
        CacheMatch { node, pages }
    }
}
impl PrefixCache for RadixCache {
    fn lookup(&mut self, tokens: &[u32]) -> CacheMatch {
        let end = tokens.len() / self.page_size * self.page_size;
        let (id, _) = self.walk(&tokens[..end]);
        self.handle(id)
    }
    fn insert(&mut self, tokens: &[u32], pages: &[PageId]) -> InsertResult {
        let end = tokens.len() / self.page_size * self.page_size;
        assert!(pages.len() >= end / self.page_size);
        let (mut id, common) = self.walk(&tokens[..end]);
        let retained = pages[common / self.page_size..end / self.page_size].to_vec();
        if common < end {
            let selector = tokens[common..common + self.page_size].to_vec();
            let parent = id;
            id = self.add(Node {
                key: tokens[common..end].to_vec(),
                pages: retained.clone(),
                parent,
                tick: self.clock,
                ..Node::default()
            });
            self.node_mut(parent).children.insert(selector, id);
        }
        InsertResult {
            handle: self.handle(id),
            retained,
        }
    }
    fn pin(&mut self, mut id: usize) {
        while id != 0 {
            let n = self.node_mut(id);
            n.pins += 1;
            id = n.parent;
        }
    }
    fn unpin(&mut self, mut id: usize) {
        while id != 0 {
            let n = self.node_mut(id);
            assert!(n.pins > 0);
            n.pins -= 1;
            id = n.parent;
        }
    }
    fn evictable_pages(&self) -> usize {
        self.nodes
            .iter()
            .flatten()
            .filter(|n| n.pins == 0)
            .map(|n| n.pages.len())
            .sum()
    }
    fn evict(&mut self, count: usize) -> Vec<PageId> {
        let mut pages = vec![];
        while pages.len() < count {
            let candidate = self
                .nodes
                .iter()
                .enumerate()
                .skip(1)
                .filter_map(|(i, n)| {
                    n.as_ref()
                        .filter(|n| n.pins == 0 && n.children.is_empty())
                        .map(|n| (i, n.tick))
                })
                .min_by_key(|&(i, t)| (t, i));
            let Some((id, _)) = candidate else { break };
            let n = self.nodes[id].take().unwrap();
            let selector = n.key[..self.page_size].to_vec();
            self.node_mut(n.parent).children.remove(&selector);
            pages.extend(n.pages);
            self.vacant.push(id);
        }
        pages
    }
    fn resident_pages(&self) -> Vec<PageId> {
        self.nodes
            .iter()
            .flatten()
            .flat_map(|n| n.pages.iter().copied())
            .collect()
    }
}

struct PagePool {
    refs: Vec<usize>,
    free: Vec<PageId>,
}
impl PagePool {
    fn new(count: usize) -> Self {
        Self {
            refs: vec![0; count],
            free: (0..count as u32).rev().collect(),
        }
    }
    fn retain(&mut self, id: PageId) {
        assert!(self.refs[id as usize] > 0);
        self.refs[id as usize] += 1;
    }
    fn release(&mut self, id: PageId) {
        let r = &mut self.refs[id as usize];
        assert!(*r > 0, "double page release");
        *r -= 1;
        if *r == 0 {
            self.free.push(id);
        }
    }
    fn allocate(&mut self) -> PageId {
        let p = self.free.pop().expect("reserved page capacity");
        assert_eq!(self.refs[p as usize], 0);
        self.refs[p as usize] = 1;
        p
    }
}
/// A lease is moved with its request and returned exactly once through release().
pub struct PageLease {
    pages: Vec<PageId>,
    cached_tokens: usize,
    node: usize,
    remaining: usize,
}
impl PageLease {
    pub fn pages(&self) -> &[PageId] {
        &self.pages
    }
    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }
}
pub struct CacheManager {
    pool: PagePool,
    cache: Box<dyn PrefixCache>,
    page_size: usize,
    reserved: usize,
}
impl CacheManager {
    pub fn new(count: usize, page_size: usize, radix: bool) -> Self {
        assert!(count > 0 && count <= u32::MAX as usize && page_size > 0);
        Self {
            pool: PagePool::new(count),
            cache: if radix {
                Box::new(RadixCache::new(page_size))
            } else {
                Box::new(NaiveCache)
            },
            page_size,
            reserved: 0,
        }
    }
    pub fn total_pages(&self) -> usize {
        self.pool.refs.len()
    }
    pub fn free_pages(&self) -> usize {
        self.pool.free.len()
    }
    pub fn cached_pages(&self) -> usize {
        self.cache.resident_pages().len()
    }
    pub fn acquire(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Option<PageLease>> {
        let total = prompt
            .len()
            .checked_add(max_tokens)
            .ok_or_else(|| Error::Invalid("sequence length overflow".into()))?
            .div_ceil(self.page_size);
        if total > self.total_pages() {
            return Err(Error::Capacity(
                "request exceeds total KV pool capacity".into(),
            ));
        }
        let handle = self.cache.lookup(&prompt[..prompt.len().saturating_sub(1)]);
        self.cache.pin(handle.node);
        let need = total - handle.pages.len();
        let available = self.free_pages() + self.cache.evictable_pages();
        if available < self.reserved + need {
            self.cache.unpin(handle.node);
            return Ok(None);
        }
        for &page in &handle.pages {
            self.pool.retain(page);
        }
        self.reserved += need;
        Ok(Some(PageLease {
            cached_tokens: handle.pages.len() * self.page_size,
            pages: handle.pages,
            node: handle.node,
            remaining: need,
        }))
    }
    pub fn extend(&mut self, lease: &mut PageLease, computed_tokens: usize) {
        let needed = computed_tokens
            .div_ceil(self.page_size)
            .saturating_sub(lease.pages.len());
        assert!(needed <= lease.remaining);
        if self.free_pages() < needed {
            let evicted = self.cache.evict(needed - self.free_pages());
            for p in evicted {
                self.pool.release(p);
            }
        }
        assert!(self.free_pages() >= needed);
        lease
            .pages
            .extend((0..needed).map(|_| self.pool.allocate()));
        lease.remaining -= needed;
        self.reserved -= needed;
    }
    pub fn publish(&mut self, lease: &mut PageLease, tokens: &[u32], computed_tokens: usize) {
        let insert = self.cache.insert(&tokens[..computed_tokens], &lease.pages);
        for p in insert.retained {
            self.pool.retain(p);
        }
        // Pin the replacement path first so overlapping paths cannot become evictable.
        self.cache.pin(insert.handle.node);
        self.cache.unpin(lease.node);
        lease.node = insert.handle.node;
    }
    pub fn release(&mut self, lease: PageLease) {
        self.cache.unpin(lease.node);
        self.reserved -= lease.remaining;
        for p in lease.pages {
            self.pool.release(p);
        }
    }
    pub fn clear_idle_cache(&mut self) {
        for p in self.cache.evict(usize::MAX) {
            self.pool.release(p);
        }
    }
    pub fn check_integrity(&self, leases: &[&PageLease]) {
        let mut expected = vec![0usize; self.total_pages()];
        for p in self.cache.resident_pages() {
            expected[p as usize] += 1;
        }
        for l in leases {
            for &p in &l.pages {
                expected[p as usize] += 1;
            }
        }
        assert_eq!(expected, self.pool.refs);
        assert_eq!(
            self.reserved,
            leases.iter().map(|l| l.remaining).sum::<usize>()
        );
        let mut seen = vec![false; self.total_pages()];
        for &p in &self.pool.free {
            assert!(!seen[p as usize]);
            seen[p as usize] = true;
        }
        for (r, free) in self.pool.refs.iter().zip(seen) {
            assert_eq!(*r == 0, free);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shares_only_full_pages_and_recomputes_last_token() {
        let mut c = CacheManager::new(32, 4, true);
        let p: Vec<u32> = (0..12).collect();
        let mut a = c.acquire(&p, 4).unwrap().unwrap();
        c.extend(&mut a, 12);
        c.publish(&mut a, &p, 12);
        let b = c.acquire(&p, 4).unwrap().unwrap();
        assert_eq!(b.cached_tokens, 8);
        assert_eq!(&a.pages[..2], &b.pages[..2]);
        c.check_integrity(&[&a, &b]);
        c.release(a);
        c.check_integrity(&[&b]);
        c.release(b);
        c.clear_idle_cache();
        c.check_integrity(&[]);
        assert_eq!(c.free_pages(), 32);
    }
    #[test]
    fn radix_split_and_pins_survive_branching() {
        let mut c = CacheManager::new(32, 2, true);
        let x = vec![1, 2, 3, 4, 5, 6];
        let y = vec![1, 2, 3, 4, 9, 8];
        let mut a = c.acquire(&x, 2).unwrap().unwrap();
        c.extend(&mut a, 6);
        c.publish(&mut a, &x, 6);
        let mut b = c.acquire(&y, 2).unwrap().unwrap();
        assert_eq!(b.cached_tokens, 4);
        c.extend(&mut b, 6);
        c.publish(&mut b, &y, 6);
        c.check_integrity(&[&a, &b]);
        c.clear_idle_cache();
        c.check_integrity(&[&a, &b]);
        c.release(a);
        c.release(b);
        c.clear_idle_cache();
        c.check_integrity(&[]);
    }
    #[test]
    fn reservations_prevent_overcommit_and_allow_eviction() {
        let mut c = CacheManager::new(4, 2, true);
        let x = vec![1, 2, 3, 4];
        let mut a = c.acquire(&x, 4).unwrap().unwrap();
        assert!(c.acquire(&[9, 8], 2).unwrap().is_none());
        c.extend(&mut a, 4);
        c.publish(&mut a, &x, 4);
        c.release(a);
        let mut b = c.acquire(&[9, 8, 7, 6], 4).unwrap().unwrap();
        c.extend(&mut b, 8);
        c.check_integrity(&[&b]);
        c.release(b);
        c.clear_idle_cache();
        c.check_integrity(&[]);
    }
    #[test]
    fn randomized_lifetimes_conserve_every_page() {
        let mut c = CacheManager::new(64, 4, true);
        let mut live: Vec<(PageLease, Vec<u32>)> = vec![];
        let mut rng = 42u64;
        for _ in 0..2000 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if rng.is_multiple_of(3) && !live.is_empty() {
                let i = rng as usize % live.len();
                let (l, _) = live.swap_remove(i);
                c.release(l);
            } else {
                let p: Vec<u32> = (0..(8 + rng as usize % 20))
                    .map(|i| if i < 8 { i as u32 } else { (rng % 7) as u32 })
                    .collect();
                if let Some(mut l) = c.acquire(&p, 8).unwrap() {
                    c.extend(&mut l, p.len());
                    c.publish(&mut l, &p, p.len());
                    live.push((l, p));
                }
            }
            c.check_integrity(&live.iter().map(|(l, _)| l).collect::<Vec<_>>());
        }
        for (l, _) in live {
            c.release(l);
        }
        c.clear_idle_cache();
        c.check_integrity(&[]);
    }
}
