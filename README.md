# Remote Ollama inference (GPU laptop)

This project includes `ollama-chat`, a Python client that sends text and optional
images to Ollama running on another laptop. The client laptop sends the request;
the laptop running Ollama performs inference.

## 1. Start Ollama on the GPU laptop

Install Ollama there, then download a vision model that fits the available
memory. For the GTX 1050 with 4 GB VRAM, start with the smaller Qwen2.5-VL 3B
tag:

```bash
ollama pull qwen2.5vl:3b
```

Ollama normally listens only on localhost. On Linux with the systemd service,
create an override so clients on your trusted LAN can reach it:

```bash
sudo systemctl edit ollama
```

Add these lines in the editor:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Save and exit, then restart Ollama:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Allow inbound TCP port `11434` only from the client laptop or trusted LAN in the
GPU laptop's firewall. Ollama's local server does not provide authentication by
default; do not expose this port directly to the public internet. Find the GPU
laptop's reachable LAN/VPN address (for example, `100.95.220.38`) and verify
that Ollama is serving:

```bash
curl http://100.95.220.38:11434/api/tags
```

Replace the example IP with the actual address. The Ollama service must be
reachable from the client laptop; `localhost` in the Python client would point
to the client itself.

If the Python client reports a connection error, run this from the client
laptop using the same address configured in `OLLAMA_HOST`:

```bash
curl http://100.95.220.38:11434/api/tags
```

If this cannot connect, check Ollama locally on the GPU laptop with
`curl http://localhost:11434/api/tags`, confirm the systemd override is active,
and allow inbound TCP port `11434` through its firewall from the client laptop.

## 2. Send a request from the client laptop

Edit `OLLAMA_HOST`, `MODEL`, `PDF_PATH`, and optionally `OUTPUT_PATH` near the top of
`src/tp3_vlm_extraction/ollama_client.py`. The script renders every page of the
PDF to an in-memory PNG and sends the page images together with the extraction
prompt. The prompt tells the VLM to collect actif and passif rows across pages;
the passif section is empty with null totals when it is absent.

Install the project and its Ollama Python client:

```bash
python -m pip install -e .
```

Run the script:

```bash
python -m tp3_vlm_extraction.ollama_client
```

The script removes an optional Markdown JSON code fence, parses the response,
normalizes French-formatted amounts and `DD/MM/YYYY` dates, then saves it to
`output/` using the PDF's basename (for example, `output/bilan_group_1.json`). If
the response isn't valid JSON, it prints the response and exits without writing
a JSON file. The Ollama request uses JSON output mode, a larger context window,
and a higher output-token limit to reduce malformed or cut-off responses.

The PDF is rendered on the client laptop, then the resulting page images are
sent to Ollama on the GPU laptop. Large PDFs with many pages can use substantial
memory and may exceed the model's image/context limits; for those, select only
the pages containing the actif table in `PDF_PATH` or process page ranges in
separate requests.
