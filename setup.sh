mkdir -p ~/.streamlit/
echo "[general]" > ~/.streamlit/credentials.toml
echo "email = \"your-email@domain.com\"" >> ~/.streamlit/credentials.toml
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "port = $PORT" >> ~/.streamlit/config.toml
