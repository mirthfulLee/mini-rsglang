// CUDA 12.6 compatibility backend. All launches are wrapped by checked Rust APIs.
#include <cuda_bf16.h>
#include <math_constants.h>
typedef __nv_bfloat16 bf16;
__device__ float f(bf16 x) { return __bfloat162float(x); }
__device__ bf16 b(float x) { return __float2bfloat16_rn(x); }
__device__ float warp_sum(float x) {
    for (int o=16;o>0;o/=2) x += __shfl_down_sync(0xffffffff,x,o);
    return x;
}
extern "C" __global__ void embedding(const bf16* w, const unsigned* ids, bf16* out, int n, int d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*d) out[i]=w[(size_t)ids[i/d]*d+i%d];
}
extern "C" __global__ void embedding_shard(const bf16* w,const unsigned* ids,bf16* out,int n,int d,unsigned start,unsigned rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*d) {unsigned id=ids[i/d];out[i]=(id>=start && id-start<rows)?w[(size_t)(id-start)*d+i%d]:b(0);}
}
extern "C" __global__ void to_float(const bf16* x,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=f(x[i]);
}
extern "C" __global__ void to_bf16(const float* x,bf16* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=b(x[i]);
}
extern "C" __global__ void gather_logits(const float* ranked,float* out,int rows,int vocab,int local_vocab) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*vocab) {int row=i/vocab,col=i%vocab;out[i]=ranked[((size_t)(col/local_vocab)*rows+row)*local_vocab+col%local_vocab];}
}
extern "C" __global__ void split_columns(const bf16* x,bf16* out,int n,int full_width,int start,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*width)out[i]=x[(size_t)(i/width)*full_width+start+i%width];
}
// Stable top-k on FP32 probabilities, matching Qwen3's BF16 routing-weight contract.
extern "C" __global__ void route(const bf16* logits,unsigned* ids,bf16* weights,int experts,int topk,int renormalize) {
    if(threadIdx.x!=0)return;
    int row=blockIdx.x;
    float maximum=-CUDART_INF_F,sum=0;
    for(int j=0;j<experts;j++)maximum=fmaxf(maximum,f(logits[(size_t)row*experts+j]));
    for(int j=0;j<experts;j++)sum+=expf(f(logits[(size_t)row*experts+j])-maximum);
    float selected=0;
    for(int slot=0;slot<topk;slot++) {
        int id=-1;float best=-CUDART_INF_F;
        for(int j=0;j<experts;j++) {
            bool used=false;for(int s=0;s<slot;s++)used|=ids[(size_t)row*topk+s]==(unsigned)j;
            float value=f(logits[(size_t)row*experts+j]);
            if(!used && (id<0 || value>best)) {id=j;best=value;}
        }
        ids[(size_t)row*topk+slot]=id;
        selected+=expf(best-maximum)/sum;
    }
    for(int slot=0;slot<topk;slot++) {
        unsigned id=ids[(size_t)row*topk+slot];
        float weight=expf(f(logits[(size_t)row*experts+id])-maximum)/sum;
        weights[(size_t)row*topk+slot]=b(renormalize?weight/selected:weight);
    }
}
extern "C" __global__ void weighted_scatter(const bf16* x,const unsigned* rows,const bf16* weights,float* out,int n,int d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    // Each expert's row list has unique destinations; expert launches are stream ordered.
    if(i<n*d) {size_t dst=(size_t)rows[i/d]*d+i%d;out[dst]+=f(b(f(x[i])*f(weights[i/d])));}
}
extern "C" __global__ void add(const bf16* x,const bf16* y,bf16* z,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) z[i]=b(f(x[i])+f(y[i]));
}
extern "C" __global__ void rms(const bf16* x,const bf16* w,bf16* out,int d,float eps) {
    __shared__ float sums[8];
    int row=blockIdx.x, t=threadIdx.x;
    float sum=0;
    for(int i=t;i<d;i+=blockDim.x) { float v=f(x[(size_t)row*d+i]); sum+=v*v; }
    sum=warp_sum(sum);
    if(t%32==0) sums[t/32]=sum;
    __syncthreads();
    if(t==0) { float total=0; for(int i=0;i<blockDim.x/32;i++)total+=sums[i]; sums[0]=rsqrtf(total/d+eps); }
    __syncthreads();
    // HF casts normalized values to BF16 before multiplying by the BF16 weight.
    for(int i=t;i<d;i+=blockDim.x) out[(size_t)row*d+i]=b(f(b(f(x[(size_t)row*d+i])*sums[0]))*f(w[i]));
}
extern "C" __global__ void rope(bf16* x,const unsigned* pos,const float* freq,int n,int heads,int d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int half=d/2;
    if(i>=n*heads*half)return;
    int j=i%half; int row=i/half;
    float angle=(float)pos[row/heads]*freq[j];
    float s,c; sincosf(angle,&s,&c); s=f(b(s)); c=f(b(c));
    size_t off=(size_t)row*d+j;
    float a=f(x[off]), z=f(x[off+half]);
    x[off]=b(f(b(a*c))-f(b(z*s)));
    x[off+half]=b(f(b(z*c))+f(b(a*s)));
}
extern "C" __global__ void swiglu(const bf16* gate,const bf16* up,bf16* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { float v=f(gate[i]); out[i]=b(f(b(v/(1.0f+expf(-v))))*f(up[i])); }
}
extern "C" __global__ void scatter(const bf16* k,const bf16* v,const unsigned* slots,bf16* kc,bf16* vc,int n,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*width) { size_t o=(size_t)slots[i/width]*width+i%width; kc[o]=k[i]; vc[o]=v[i]; }
}
// One CTA per query/head. Online softmax across 32-key tiles, FP32 accumulation.
// Prefill and decode use the same causal paged addressing, including cached prefixes.
extern "C" __global__ void attention(const bf16* q,const bf16* kc,const bf16* vc,
    const unsigned* pos,const unsigned* seq,const unsigned* table,bf16* out,
    int qheads,int kvheads,int d,int page_size,int table_width,float scale_q) {
    __shared__ float query[256], scores[32], probs[32], alpha, denom, maximum;
    int row=blockIdx.x, head=blockIdx.y, t=threadIdx.x, lane=t%32, warp=t/32;
    int kh=head/(qheads/kvheads), length=pos[row]+1;
    const unsigned* pages=table+(size_t)seq[row]*table_width;
    for(int j=t;j<d;j+=128)query[j]=f(q[((size_t)row*qheads+head)*d+j])*scale_q;
    if(t==0) { denom=0; maximum=-CUDART_INF_F; }
    float acc[2]={0,0};
    __syncthreads();
    for(int base=0;base<length;base+=32) {
        for(int j=warp;j<32;j+=4) {
            int p=base+j;
            float dot=0;
            if(p<length) {
                size_t off=((size_t)pages[p/page_size]*page_size+p%page_size)*kvheads*d+(size_t)kh*d;
                for(int z=lane;z<d;z+=32)dot+=query[z]*(f(kc[off+z])*scale_q);
            }
            dot=warp_sum(dot);
            if(lane==0)scores[j]=p<length ? dot : -CUDART_INF_F;
        }
        __syncthreads();
        if(t==0) {
            float m=maximum;
            for(int j=0;j<32;j++)m=fmaxf(m,scores[j]);
            alpha=expf(maximum-m);
            float sum=0;
            for(int j=0;j<32;j++) { probs[j]=expf(scores[j]-m); sum+=probs[j]; }
            denom=denom*alpha+sum; maximum=m;
        }
        __syncthreads();
        for(int z=t;z<d;z+=128) {
            float v=0;
            for(int j=0;j<32 && base+j<length;j++) {
                int p=base+j;
                size_t off=((size_t)pages[p/page_size]*page_size+p%page_size)*kvheads*d+(size_t)kh*d+z;
                v+=probs[j]*f(vc[off]);
            }
            acc[z/128]=acc[z/128]*alpha+v;
        }
        __syncthreads();
    }
    for(int z=t;z<d;z+=128)out[((size_t)row*qheads+head)*d+z]=b(acc[z/128]/denom);
}
extern "C" __global__ void gather(const bf16* x,const unsigned* rows,bf16* out,int n,int d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*d)out[i]=x[(size_t)rows[i/d]*d+i%d];
}
extern "C" __global__ void argmax(const float* x,unsigned* out,int d) {
    __shared__ float scores[256]; __shared__ unsigned ids[256];
    int t=threadIdx.x; float best=-CUDART_INF_F; unsigned id=0;
    for(int j=t;j<d;j+=256) { float v=x[(size_t)blockIdx.x*d+j]; if(v>best || (v==best && (unsigned)j<id)){ best=v; id=j; } }
    scores[t]=best; ids[t]=id; __syncthreads();
    for(int s=128;s>0;s/=2) {
        if(t<s && (scores[t+s]>scores[t] || (scores[t+s]==scores[t] && ids[t+s]<ids[t]))) { scores[t]=scores[t+s]; ids[t]=ids[t+s]; }
        __syncthreads();
    }
    if(t==0)out[blockIdx.x]=ids[0];
}

// Reproducible FP32 all-reduce: all-gather transport, fixed rank order per element.
extern "C" __global__ void sum_ranks(const float* ranked,float* out,int n,int ranks) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float sum=0;for(int rank=0;rank<ranks;rank++)sum+=ranked[(size_t)rank*n+i];out[i]=sum;}
}
