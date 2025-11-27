vllm serve /path/to/your/model  \
--port 8000 \
--trust-remote-code \
--disable-log-requests \
--max-model-len 32768 \
--gpu-memory-utilization 0.8 \
--tensor-parallel-size 8
