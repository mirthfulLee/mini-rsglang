use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use rsglang_core::*;
use rsglang_runtime::{ChatMessage, EngineHandle};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn router(engine: EngineHandle) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .with_state(engine)
}
struct ApiError(Error);
impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        Self(e)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = match self.0 {
            Error::Invalid(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Error::Capacity(_) => (StatusCode::TOO_MANY_REQUESTS, "capacity_error"),
            Error::Stopped => (StatusCode::SERVICE_UNAVAILABLE, "engine_stopped"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "backend_error"),
        };
        (status,Json(json!({"error":{"message":self.0.to_string(),"type":kind,"param":null,"code":null}}))).into_response()
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApiRequest {
    model: Option<String>,
    prompt: Option<Value>,
    messages: Option<Vec<ChatMessage>>,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "one")]
    top_p: f32,
    top_k: Option<usize>,
    #[serde(default = "max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default = "single")]
    n: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    enable_thinking: bool,
    stream_options: Option<StreamOptions>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}
fn one() -> f32 {
    1.
}
fn single() -> usize {
    1
}
fn max_tokens() -> usize {
    128
}
async fn health(State(e): State<EngineHandle>) -> Response {
    (
        if e.healthy() {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({"status":if e.healthy(){"ready"}else{"stopped"}})),
    )
        .into_response()
}
async fn metrics(State(e): State<EngineHandle>) -> Response {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        e.prometheus_metrics(),
    )
        .into_response()
}
async fn models(State(e): State<EngineHandle>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":e.info().model_id,"object":"model","created":0,"owned_by":"rsglang"}]}),
    )
}
async fn completions(
    State(e): State<EngineHandle>,
    body: std::result::Result<Json<ApiRequest>, JsonRejection>,
) -> std::result::Result<Response, ApiError> {
    generate(e, body, false).await
}
async fn chat(
    State(e): State<EngineHandle>,
    body: std::result::Result<Json<ApiRequest>, JsonRejection>,
) -> std::result::Result<Response, ApiError> {
    generate(e, body, true).await
}
fn reason(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Error => "error",
    }
}
async fn generate(
    e: EngineHandle,
    body: std::result::Result<Json<ApiRequest>, JsonRejection>,
    chat: bool,
) -> std::result::Result<Response, ApiError> {
    let Json(req) = body.map_err(|err| ApiError(Error::Invalid(err.body_text())))?;
    if req.n != 1 {
        return Err(Error::Invalid("only n=1 is supported".into()).into());
    }
    if req.model.as_ref().is_some_and(|m| m != &e.info().model_id) {
        return Err(Error::Invalid("model does not match the loaded model".into()).into());
    }
    if !req.stream && req.stream_options.is_some() {
        return Err(Error::Invalid("stream_options requires stream=true".into()).into());
    }
    let prompt = if chat {
        if req.prompt.is_some() {
            return Err(Error::Invalid("chat accepts messages, not prompt".into()).into());
        }
        Prompt::Text(
            e.text().chat(
                req.messages
                    .as_deref()
                    .ok_or_else(|| Error::Invalid("missing messages".into()))?,
                req.enable_thinking,
            )?,
        )
    } else {
        if req.messages.is_some() || req.enable_thinking {
            return Err(
                Error::Invalid("messages/enable_thinking require chat endpoint".into()).into(),
            );
        }
        match req
            .prompt
            .ok_or_else(|| Error::Invalid("missing prompt".into()))?
        {
            Value::String(s) => Prompt::Text(s),
            Value::Array(ids) => Prompt::TokenIds(
                ids.into_iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .ok_or_else(|| {
                                Error::Invalid("prompt array must contain token IDs".into())
                            })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => {
                return Err(Error::Invalid("prompt must be text or a token ID array".into()).into())
            }
        }
    };
    let sampling = SamplingParams {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        seed: req.seed,
        max_tokens: req.max_tokens,
        ignore_eos: req.ignore_eos,
    };
    let mut events = e.generate(GenerateRequest { prompt, sampling }).await?;
    let id = format!(
        "{}-{}",
        if chat { "chatcmpl" } else { "cmpl" },
        events.request_id
    );
    let model = e.info().model_id.clone();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if req.stream {
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        let stream = async_stream::stream! {
            if chat {yield Ok::<_,Infallible>(Event::default().json_data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]})).unwrap());}
            let mut terminal=false;
            while let Some(event)=events.recv().await {
                let payload=match event {
                    GenerationEvent::Token{text,..}|GenerationEvent::Text{text,..}=>{
                        let choice=if chat{json!({"index":0,"delta":{"content":text},"finish_reason":null})}else{json!({"index":0,"text":text,"finish_reason":null,"logprobs":null})};
                        json!({"id":id,"object":if chat{"chat.completion.chunk"}else{"text_completion"},"created":created,"model":model,"choices":[choice]})
                    },
                    GenerationEvent::Finished{reason:r,prompt_tokens,completion_tokens,cached_tokens,..}=>{
                        let choice=if chat{json!({"index":0,"delta":{},"finish_reason":reason(r)})}else{json!({"index":0,"text":"","finish_reason":reason(r),"logprobs":null})};
                        yield Ok(Event::default().json_data(json!({"id":id,"object":if chat{"chat.completion.chunk"}else{"text_completion"},"created":created,"model":model,"choices":[choice]})).unwrap());
                        if include_usage {yield Ok(Event::default().json_data(json!({"id":id,"object":if chat{"chat.completion.chunk"}else{"text_completion"},"created":created,"model":model,"choices":[],"usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"total_tokens":prompt_tokens+completion_tokens,"prompt_tokens_details":{"cached_tokens":cached_tokens}}})).unwrap());}
                        terminal=true;break;
                    },
                    GenerationEvent::Error{message,..}=>{yield Ok(Event::default().json_data(json!({"error":{"message":message,"type":"backend_error"}})).unwrap());terminal=true;break;},
                };
                yield Ok(Event::default().json_data(payload).unwrap());
            }
            if !terminal {yield Ok(Event::default().json_data(json!({"error":{"message":"generation stream closed before completion","type":"backend_error"}})).unwrap());}
            yield Ok(Event::default().data("[DONE]"));
        };
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let mut text = String::new();
        while let Some(event) = events.recv().await {
            match event {
                GenerationEvent::Token { text: delta, .. }
                | GenerationEvent::Text { text: delta, .. } => text.push_str(&delta),
                GenerationEvent::Error { message, .. } => {
                    return Err(Error::Backend(message).into())
                }
                GenerationEvent::Finished {
                    reason: r,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    ..
                } => {
                    if !matches!(r, FinishReason::Length | FinishReason::Stop) {
                        return Err(
                            Error::Backend(format!("generation ended: {}", reason(r))).into()
                        );
                    }
                    let choice = if chat {
                        json!({"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason(r)})
                    } else {
                        json!({"index":0,"text":text,"finish_reason":reason(r),"logprobs":null})
                    };
                    return Ok(Json(json!({"id":id,"object":if chat{"chat.completion"}else{"text_completion"},"created":created,"model":model,"choices":[choice],"usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"total_tokens":prompt_tokens+completion_tokens,"prompt_tokens_details":{"cached_tokens":cached_tokens}}})).into_response());
                }
            }
        }
        Err(Error::Backend("generation stream closed before completion".into()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_parameters_are_rejected() {
        for field in [
            "tools",
            "logprobs",
            "stop",
            "frequency_penalty",
            "response_format",
        ] {
            let v = json!({"prompt":"hello",field:null});
            assert!(serde_json::from_value::<ApiRequest>(v).is_err(), "{field}");
        }
        assert!(serde_json::from_value::<ApiRequest>(
            json!({"messages":[{"role":"user","content":[{"type":"image_url"}]}]})
        )
        .is_err());
    }
    #[test]
    fn defaults_and_stream_options() {
        let r: ApiRequest = serde_json::from_value(json!({"prompt":"hello"})).unwrap();
        assert_eq!(r.max_tokens, 128);
        assert_eq!(r.n, 1);
        assert_eq!(r.temperature, 0.);
        assert_eq!(r.top_p, 1.);
        let r: ApiRequest = serde_json::from_value(
            json!({"prompt":[1,2],"stream":true,"stream_options":{"include_usage":true}}),
        )
        .unwrap();
        assert!(r.stream_options.unwrap().include_usage);
    }
}
