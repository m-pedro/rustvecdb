use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use serde::{Deserialize, Serialize};

use crate::collection::Collection;
use crate::error::Result;
use crate::metadata::{Filter, Metadata};

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub id: u64,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub exact: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub id: u64,
}

pub fn serve_http(addr: &str, collection: Collection) -> Result<()> {
    let listener = TcpListener::bind(addr)?;
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let col = collection.clone();
        thread::spawn(move || {
            let _ = handle_connection(stream, col);
        });
    }
    Ok(())
}

pub fn handle_connection(mut stream: TcpStream, collection: Collection) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }

    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return write_json_response(&mut stream, 400, r#"{"error":"malformed request line"}"#);
    }
    let (method, path) = (parts[0], parts[1]);

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse::<usize>().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }

    match (method, path) {
        ("GET", "/stats") => {
            let stats = collection.stats();
            let json = serde_json::to_string(&stats)?;
            write_json_response(&mut stream, 200, &json)
        }
        ("POST", "/upsert") => match serde_json::from_slice::<UpsertRequest>(&body) {
            Ok(req) => match collection.upsert(req.id, req.vector, req.metadata) {
                Ok(()) => write_json_response(&mut stream, 200, r#"{"status":"ok"}"#),
                Err(e) => write_json_response(
                    &mut stream,
                    400,
                    &serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            },
            Err(e) => write_json_response(
                &mut stream,
                400,
                &serde_json::json!({ "error": e.to_string() }).to_string(),
            ),
        },
        ("POST", "/query") => match serde_json::from_slice::<QueryRequest>(&body) {
            Ok(req) => {
                let res = if req.exact {
                    collection.search_exact(&req.vector, req.k, req.filter.as_ref())
                } else {
                    collection.search(&req.vector, req.k, req.filter.as_ref())
                };
                match res {
                    Ok(hits) => {
                        let json = serde_json::to_string(&serde_json::json!({ "results": hits }))?;
                        write_json_response(&mut stream, 200, &json)
                    }
                    Err(e) => write_json_response(
                        &mut stream,
                        400,
                        &serde_json::json!({ "error": e.to_string() }).to_string(),
                    ),
                }
            }
            Err(e) => write_json_response(
                &mut stream,
                400,
                &serde_json::json!({ "error": e.to_string() }).to_string(),
            ),
        },
        ("POST", "/delete") => match serde_json::from_slice::<DeleteRequest>(&body) {
            Ok(req) => match collection.delete(req.id) {
                Ok(removed) => write_json_response(
                    &mut stream,
                    200,
                    &serde_json::json!({ "deleted": removed }).to_string(),
                ),
                Err(e) => write_json_response(
                    &mut stream,
                    400,
                    &serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            },
            Err(e) => write_json_response(
                &mut stream,
                400,
                &serde_json::json!({ "error": e.to_string() }).to_string(),
            ),
        },
        _ => write_json_response(&mut stream, 404, r#"{"error":"not found"}"#),
    }
}

fn write_json_response(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        reason,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::hnsw::HnswParams;

    fn send_raw_http(addr: &str, method: &str, path: &str, body: &str) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        let req = format!(
            "{} {} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            method,
            path,
            body.len(),
            body
        );
        stream.write_all(req.as_bytes()).unwrap();
        let mut resp = String::new();
        stream.read_to_string(&mut resp).unwrap();
        resp.split_once("\r\n\r\n").unwrap().1.to_string()
    }

    #[test]
    fn test_http_server_upsert_query_and_stats() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let col = Collection::new_in_memory(3, Metric::L2, HnswParams::default()).unwrap();

        let server_col = col.clone();
        thread::spawn(move || {
            for _ in 0..3 {
                if let Ok((stream, _)) = listener.accept() {
                    let _ = handle_connection(stream, server_col.clone());
                }
            }
        });

        let r1 = send_raw_http(
            &addr,
            "POST",
            "/upsert",
            r#"{"id":7,"vector":[1.0,2.0,3.0],"metadata":{}}"#,
        );
        assert!(r1.contains(r#""status":"ok""#));

        let r2 = send_raw_http(
            &addr,
            "POST",
            "/query",
            r#"{"vector":[1.0,2.0,3.1],"k":1}"#,
        );
        assert!(r2.contains(r#""id":7"#));

        let r3 = send_raw_http(&addr, "GET", "/stats", "");
        assert!(r3.contains(r#""vector_count":1"#));
    }
}
