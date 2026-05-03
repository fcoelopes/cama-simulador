from google.generativeai.models import list_models
from google.generativeai.client import configure

# Tente mudar para 'v1' se o 'v1beta' continuar dando 404
configure(api_key="AIzaSyDzLE6__suXvc_QPDItcU9zHBOC9CAsUpo", transport="rest")

print("Modelos disponíveis para generateContent:")
try:
    for m in list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Nome: {m.name} | Versão: {m.version if hasattr(m, 'version') else 'N/A'}")
except Exception as e:
    print(f"Erro: {e}")