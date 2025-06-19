# Sistema Personalizado Flipped Classroom con Gemini y Streamlit

Sistema interactivo para aprendizaje personalizado basado en Flipped Classroom, usando Streamlit y la API Gemini de Google para generar contenido educativo dinámico según el nivel del estudiante.

---

## Características

- Actualiza niveles de conocimiento desde CSV de exámenes.
- Genera contenido adaptado a cada estudiante y concepto.
- Visualiza niveles con gráficos interactivos.
- Permite actualización manual de niveles para administradores.

---

## Requisitos

- Python 3.8+
- Paquetes: `streamlit`, `pandas`, `altair`, `google-generativeai`
- API Key de Gemini configurada en Streamlit Secrets o variable de entorno `GEMINI_API_KEY`
- Archivo `estudiantes.csv` con estructura adecuada.

---

## Instalación

```bash
pip install streamlit pandas altair google-generativeai
Uso
Ejecutar:

bash
Copiar
Editar
streamlit run app.py
Subir CSV con resultados de examen para actualizar niveles.

Ingresar ID de estudiante para contenido personalizado.

Administradores pueden actualizar niveles manualmente.

Archivo estudiantes.csv
Debe contener columnas:
id, nombre, FC_DEFINICION, FC_ROLES, FC_TECNOLOGIA, FC_APLICACION, FC_BENEFICIOS
con niveles (0.25, 0.55, 0.85) por concepto.

Configuración API
Configurar clave en ~/.streamlit/secrets.toml o variable GEMINI_API_KEY.

Futuras mejoras
Integrar RAG para base de conocimiento.

Autenticación y seguridad.

Mejoras UI/UX y despliegue en nube.
