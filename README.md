<p align="center">
  <img src="https://svg-banners.vercel.app/api?type=origin&text1=Proyecto%20IA%20Multiagente&text2=Juan%20Esteban%20y%20Sebasti%C3%A1n&width=1000&height=250" alt="Proyecto IA Multiagente">
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/LangChain-Multiagent-orange?style=for-the-badge&logo=robotframework">
  <img src="https://img.shields.io/badge/Streamlit-App-green?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/AI-RAG%20System-purple?style=for-the-badge&logo=github">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge&logo=progress">
</p>

✨ Descripción General

Este proyecto implementa un sistema avanzado de Generación Aumentada por Recuperación (RAG) utilizando un enfoque Multiagente para mejorar la precisión y contextualización de las respuestas.

Utiliza el modelo de lenguaje de Google Gemini para la extracción de información, el análisis complejo y la generación final de respuestas.

El objetivo principal es proporcionar una herramienta inteligente para la consulta y análisis de grandes volúmenes de documentos (PDFs, TXT), superando las limitaciones de los sistemas RAG tradicionales al delegar tareas a agentes especializados.

🚀 Características Principales

Arquitectura Multiagente:
Sistema compuesto por agentes especializados (Extracción, Análisis, Respuesta) que colaboran para lograr resultados precisos.

Integración con Gemini:
Conexión vía API para procesamiento de lenguaje natural.

Base de Conocimiento FAISS:
Utilizada como vector store para búsqueda semántica rápida.

Interfaz Interactiva en Streamlit:
Permite al usuario interactuar fácilmente con el sistema.

🛠 Tecnologías y Requisitos
Tecnologías Clave

LLM: Gemini (vía API)

Vector Store: FAISS

Frontend: Streamlit

Lenguaje: Python 3.10+

Contenedores: Docker/Kubernetes (opcional)

⚙ Instalación y Configuración del Entorno
(Paso a paso completo para ejecutar el proyecto desde cero)
⿡ Clonar el repositorio
git clone https://github.com/Baljeet-codes/minirag-multiagente-gemini.git
cd minirag-multiagente-gemini

⿢ Crear y activar el entorno virtual
🔹 Windows (PowerShell o CMD)
python -m venv venv
venv\Scripts\activate


Si no funciona:

py -m venv venv
venv\Scripts\activate

🔹 Linux / macOS
python3 -m venv venv
source venv/bin/activate

⿣ Instalar dependencias
pip install -r requirements.txt

⿤ Configurar la clave de Gemini (.env)

Crear un archivo .env en la raíz del proyecto:

GEMINI_API_KEY="TU_CLAVE_AQUI"


El .env ya está incluido en .gitignore para evitar filtrar claves.

⿥ Ejecutar la aplicación con Streamlit

Con el entorno virtual activado:

Windows
python -m streamlit run src/app.py


o:

py -m streamlit run src/app.py

Linux / macOS
python3 -m streamlit run src/app.py


Se abrirá en tu navegador en:

👉 http://localhost:8501

⿦ Detener la aplicación

En la terminal:

CTRL + C


Desactivar entorno virtual:

deactivate

⿧ (Opcional) Despliegue en la nube

Puedes desplegar en:

Streamlit Community Cloud

Hugging Face Spaces

Conectando tu repo GitHub:
👉 https://github.com/Baljeet-codes/minirag-multiagente-gemini

▶ Uso

Sube un documento (PDF, imagen o TXT).

El Agente de Extracción limpia, transforma y divide en chunks.

El Agente de Análisis genera embeddings y busca fragmentos relevantes (distancia euclidiana).

El Agente de Respuesta genera la respuesta final usando Gemini con temperatura baja (0.0).

🤝 Contribuciones

Haz un fork del repositorio.

Crea una rama:

git checkout -b feature/nueva-caracteristica


Realiza cambios:

git commit -m "feat: Añadir nueva característica"


Sube la rama:

git push origin feature/nueva-caracteristica


Crea un Pull Request explicando los cambios.

📧 Contacto

GitHub: https://github.com/Baljeet-codes

Correo: esteban.aguirre@utp.edu.co 
s.mogollon@utp.edu.co
