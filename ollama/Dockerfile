FROM ollama/ollama:latest
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 11434
ENTRYPOINT ["/entrypoint.sh"]
