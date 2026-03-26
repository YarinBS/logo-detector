FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app

RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["bash", "-lc", "uv run python src/main.py --mode serve & uv run streamlit run src/streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]