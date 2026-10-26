//! Executes extension provider HTTP requests through the trusted host boundary.

use std::collections::BTreeMap;
use std::io::Read;

use rpi_ai::Model;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Executes one host-mediated provider HTTP request and computes its provenance.
///
/// Workers that use this function must run with direct network access denied.
/// The request is restricted to POST JSON on the selected model's origin,
/// redirects are disabled, and incomplete request identities fail closed.
///
/// # Parameters
/// - `model`: Host-selected model bound to the provider invocation.
/// - `request_json`: Versioned prepared HTTP request envelope.
///
/// # Returns
/// JSON containing HTTP status, JSON response body, and a host-generated report.
///
/// # Errors
/// Returns a policy, decoding, transport, size, or provenance error.
pub fn execute_provider_http_request(model: &Model, request_json: &str) -> Result<String, String> {
    if request_json.len() > MAX_REQUEST_BYTES {
        return Err("provider HTTP request exceeds the size limit".to_owned());
    }
    let request: PreparedProviderHttpRequest = serde_json::from_str(request_json)
        .map_err(|_| "provider HTTP request is invalid".to_owned())?;
    if request.schema_version != "rpi-provider-http-request-v1" || request.method != "POST" {
        return Err("provider HTTP request version or method is unsupported".to_owned());
    }
    let url = url::Url::parse(&request.url)
        .map_err(|_| "provider HTTP request URL is invalid".to_owned())?;
    let model_url = url::Url::parse(&model.base_url)
        .map_err(|_| "selected model endpoint is invalid".to_owned())?;
    if !same_origin(&url, &model_url)
        || url.username() != ""
        || url.password().is_some()
        || !model_url.username().is_empty()
        || model_url.password().is_some()
        || model_url.query().is_some()
        || model_url.fragment().is_some()
    {
        return Err("provider HTTP request is outside the selected model origin".to_owned());
    }

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|_| "provider HTTP transport is unavailable".to_owned())?;
    let mut headers = reqwest::header::HeaderMap::new();
    let mut canonical_headers = BTreeMap::new();
    for (name, value) in &request.headers {
        let normalized_name = name.to_ascii_lowercase();
        if canonical_headers
            .insert(normalized_name, value.clone())
            .is_some()
        {
            return Err("provider HTTP request contains duplicate headers".to_owned());
        }
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "host"
                | "content-length"
                | "transfer-encoding"
                | "connection"
                | "proxy-authorization"
                | "proxy-connection"
        ) {
            return Err("provider HTTP request contains a transport-controlled header".to_owned());
        }
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| "provider HTTP header name is invalid".to_owned())?;
        let value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|_| "provider HTTP header value is invalid".to_owned())?;
        headers.insert(name, value);
    }
    if !headers.contains_key(reqwest::header::CONTENT_TYPE) {
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        canonical_headers.insert("content-type".to_owned(), "application/json".to_owned());
    }
    if !headers.contains_key(reqwest::header::USER_AGENT) {
        headers.insert(
            reqwest::header::USER_AGENT,
            reqwest::header::HeaderValue::from_static("rpi-provider-worker/1"),
        );
        canonical_headers.insert("user-agent".to_owned(), "rpi-provider-worker/1".to_owned());
    }
    let applied_settings = rpi_ai::applied_settings_for_prepared_http_request(
        model,
        &request.url,
        &request.body,
        &canonical_headers,
    );
    if !applied_settings.complete {
        return Err("provider HTTP request cannot be fingerprinted completely".to_owned());
    }
    let response = client
        .post(request.url)
        .headers(headers)
        .body(
            serde_json::to_vec(&request.body)
                .map_err(|_| "provider HTTP request body is invalid".to_owned())?,
        )
        .send()
        .map_err(|_| "provider HTTP request failed".to_owned())?;
    let status = response.status().as_u16();
    let mut body = Vec::new();
    response
        .take((MAX_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut body)
        .map_err(|_| "provider HTTP response could not be read".to_owned())?;
    if body.len() > MAX_RESPONSE_BYTES {
        return Err("provider HTTP response exceeds the size limit".to_owned());
    }
    let body: Value = serde_json::from_slice(&body)
        .map_err(|_| "provider HTTP response is not valid JSON".to_owned())?;
    serde_json::to_string(&json!({
        "status": status,
        "body": body,
        "appliedSettings": applied_settings,
    }))
    .map_err(|_| "provider HTTP response could not be encoded".to_owned())
}

/// Holds the exact JSON request values submitted by a provider extension.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PreparedProviderHttpRequest {
    /// Versioned host request contract.
    schema_version: String,
    /// HTTP method, restricted to POST.
    method: String,
    /// Fully resolved URL within the selected model origin.
    url: String,
    /// Exact request header values.
    headers: BTreeMap<String, String>,
    /// Exact JSON body sent by the host transport.
    body: Value,
}

/// Reports whether two URLs have the same scheme, host, and effective port.
///
/// # Parameters
/// - `request`: Prepared provider request URL.
/// - `model`: Host-selected model endpoint URL.
///
/// # Returns
/// `true` when both URLs target the same origin.
fn same_origin(request: &url::Url, model: &url::Url) -> bool {
    request.scheme() == model.scheme()
        && request.host_str() == model.host_str()
        && request.port_or_known_default() == model.port_or_known_default()
}

#[cfg(test)]
mod tests {
    use super::execute_provider_http_request;
    use rpi_ai::{Api, Model};
    use serde_json::Value;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Sends one exact prepared request through the host and records its effects.
    #[test]
    fn broker_reports_the_exact_headers_and_body_sent() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local provider");
        let address = listener.local_addr().expect("read local provider address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept provider request");
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                .expect("set provider read timeout");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let count = stream.read(&mut buffer).expect("read provider request");
                request.extend_from_slice(&buffer[..count]);
                if let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or_default();
                    if request.len() >= header_end + 4 + content_length {
                        break;
                    }
                }
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 11\r\nconnection: close\r\n\r\n{\"ok\":true}")
                .expect("write provider response");
            request
        });
        let endpoint = format!("http://{address}/v1/chat/completions");
        let model = Model::new(
            "test-model",
            "Test model",
            Api::OpenaiCompletions,
            "test",
            &endpoint,
        );
        let request = serde_json::json!({
            "schemaVersion": "rpi-provider-http-request-v1",
            "method": "POST",
            "url": endpoint,
            "headers": {"authorization": "Bearer credential-sentinel"},
            "body": {"model": "test-model", "input": "prompt-sentinel"}
        });
        let result = execute_provider_http_request(
            &model,
            &serde_json::to_string(&request).expect("encode prepared request"),
        )
        .expect("brokered request succeeds");
        let captured = server.join().expect("provider server completes");
        let captured_text = String::from_utf8_lossy(&captured).to_ascii_lowercase();
        assert!(captured_text.contains("content-type: application/json"));
        assert!(captured_text.contains("user-agent: rpi-provider-worker/1"));
        assert!(captured_text.contains("authorization: bearer credential-sentinel"));
        assert!(captured_text.contains("prompt-sentinel"));
        let result: Value = serde_json::from_str(&result).expect("decode broker result");
        assert_eq!(result["body"]["ok"], true);
        assert_eq!(result["appliedSettings"]["complete"], true);
        assert_eq!(
            result["appliedSettings"]["schemaVersion"],
            "rpi-applied-request-settings-v5"
        );
        let safe_result = serde_json::to_string(&result).expect("encode safe result");
        assert!(!safe_result.contains("credential-sentinel"));
        assert!(!safe_result.contains("prompt-sentinel"));
    }

    /// Rejects a transport-controlled header before making a network request.
    #[test]
    fn broker_rejects_transport_controlled_headers() {
        let model = Model::new(
            "test-model",
            "Test model",
            Api::OpenaiCompletions,
            "test",
            "http://127.0.0.1:1/v1/chat/completions",
        );
        let request = serde_json::json!({
            "schemaVersion": "rpi-provider-http-request-v1",
            "method": "POST",
            "url": model.base_url,
            "headers": {"host": "attacker.invalid"},
            "body": {}
        });
        let error = execute_provider_http_request(
            &model,
            &serde_json::to_string(&request).expect("encode prepared request"),
        )
        .expect_err("host header must be denied");
        assert!(error.contains("transport-controlled"));
    }
}
