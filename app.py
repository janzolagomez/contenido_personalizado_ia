# -*- coding: utf-8 -*-
"""app

Este script implementa un Sistema de Aprendizaje Personalizado para Flipped Classroom
utilizando Streamlit, Pandas, la API de Gemini para la generación dinámica de contenido
y Pinecone para Retrieval Augmented Generation (RAG).
Los niveles de conocimiento de los estudiantes se gestionan a través de un archivo CSV local,
el cual puede ser actualizado manualmente o mediante la carga de resultados de exámenes (CSV).
"""

import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai
import os
import time # Para reintentos en la generación de embeddings
from pinecone import Pinecone # Importar Pinecone
# Si vas a usar LangChain para dividir texto dentro de la app (no recomendado para este caso),
# o si quieres mantener la coherencia de imports, pero no es estrictamente necesario aquí
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuración de la API de Gemini y Pinecone ---
gemini_api_key = None
pinecone_api_key = None
pinecone_environment = None
pinecone_index_name = "aulainvertigarag" # Asegúrate que este sea el nombre de tu índice

try:
    # Intenta cargar desde Streamlit Secrets (preferido para despliegue)
    gemini_api_key = st.secrets["gemini_api_key"]
    pinecone_api_key = st.secrets["pinecone_api_key"]
    pinecone_environment = st.secrets["pinecone_environment"]
    # Si el nombre del índice también es un secreto
    pinecone_index_name = st.secrets.get("pinecone_index_name", "aulainvertigarag")
except AttributeError:
    # Si no estás en Streamlit Cloud o secrets.toml no está configurado, usa variables de entorno
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "aulainvertigarag")

# Validar que las claves estén configuradas
if not gemini_api_key:
    st.error(
        "Error: La clave de API de Gemini no ha sido configurada. "
        "Asegúrate de establecer 'gemini_api_key' en Streamlit Secrets o como variable de entorno (GEMINI_API_KEY)."
    )
    st.stop()

if not pinecone_api_key or not pinecone_environment:
    st.error(
        "Error: Las claves de API o el entorno de Pinecone no han sido configurados. "
        "Asegúrate de establecer 'pinecone_api_key' y 'pinecone_environment' en Streamlit Secrets o como variables de entorno."
    )
    st.stop()

# Configurar Gemini
genai.configure(api_key=gemini_api_key)
# Considera usar 'gemini-pro' para la generación de texto de mayor calidad en RAG
model = genai.GenerativeModel(model_name="gemini-pro")
EMBEDDING_MODEL = "models/embedding-001" # El mismo modelo que usaste para indexar

# Inicializar Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
    if pinecone_index_name not in pc.list_indexes():
        st.error(f"Error: El índice de Pinecone '{pinecone_index_name}' no existe en tu entorno.")
        st.stop()
    index = pc.Index(pinecone_index_name)
    st.success(f"Conexión a Pinecone establecida con el índice: **{pinecone_index_name}**.")
except Exception as e:
    st.error(f"Error al conectar con Pinecone: {e}. Por favor, verifica tu clave API, entorno y el nombre del índice.")
    st.stop()


# --- Definiciones Globales ---
conceptos = [
    "FC_DEFINICION",
    "FC_ROLES",
    "FC_TECNOLOGIA",
    "FC_APLICACION",
    "FC_BENEFICIOS",
]

nivel_map = {
    0.25: "Básico",
    0.55: "Intermedio",
    0.85: "Avanzado",
}

# Diccionario para nombres legibles de conceptos
conceptos_legibles = {
    "FC_DEFINICION": "la definición de Flipped Classroom",
    "FC_ROLES": "los roles del estudiante y el docente en Flipped Classroom",
    "FC_TECNOLOGIA": "la tecnología usada en Flipped Classroom",
    "FC_APLICACION": "cómo aplicar Flipped Classroom",
    "FC_BENEFICIOS": "los beneficios de Flipped Classroom",
}

# --- Cargar datos de estudiantes y citas ---
# Asegúrate de que estos archivos estén en la misma carpeta que app.py o accesible
try:
    estudiantes_df = pd.read_csv("estudiantes.csv")
except FileNotFoundError:
    st.error("Error: 'estudiantes.csv' no encontrado. Asegúrate de que esté en la misma carpeta.")
    st.stop()

citations_dict = {}
try:
    citation_data_path = "Fuentes Aula invertida.csv" # Asegúrate que este archivo esté accesible
    simplified_df = pd.read_csv(citation_data_path, encoding='latin1', delimiter=';')
    # Ajusta los nombres de las columnas según tu CSV si son diferentes
    simplified_df = simplified_df.rename(columns={
        simplified_df.columns[0]: "filename",
        simplified_df.columns[1]: "cite_apa",
        simplified_df.columns[2]: "reference_apa"
    })
    for _, row in simplified_df.iterrows():
        file_name = str(row['filename']).strip()
        citations_dict[file_name] = {
            "cite_apa": str(row['cite_apa']).strip(),
            "reference_apa": str(row['reference_apa']).strip()
        }
except Exception as e:
    st.warning(f"No se pudieron cargar los metadatos de citas desde 'Fuentes Aula invertida.csv': {e}. Las citas pueden no aparecer en el contenido generado.")


# --- Funciones Auxiliares ---
def nota_a_nivel(nota):
    if nota >= 85:
        return 0.85
    elif nota >= 55:
        return 0.55
    else:
        return 0.25

def actualizar_niveles_desde_csv_examen(estudiantes_df_local, uploaded_file):
    if uploaded_file is not None:
        try:
            exam_results_df = pd.read_csv(uploaded_file)
            st.subheader("Resultados del Examen Subido (Últimas Entradas):")
            st.dataframe(exam_results_df.tail(5))

            for index, row in exam_results_df.iterrows():
                estudiante_id = row.get("ID de Estudiante", None)
                nota_examen_general = row.get("Puntuación total", None)

                if pd.notna(estudiante_id) and pd.notna(nota_examen_general):
                    estudiante_id = int(estudiante_id)
                    nota_examen_general = int(nota_examen_general)

                    nuevo_nivel_general = nota_a_nivel(nota_examen_general)
                    idx = estudiantes_df_local[estudiantes_df_local["id"] == estudiante_id].index
                    if not idx.empty:
                        for concepto in conceptos:
                            estudiantes_df_local.loc[idx, concepto] = nuevo_nivel_general
                        st.write(
                            f"Nivel de estudiante **{estudiante_id}** actualizado a **{nivel_map[nuevo_nivel_general]}** basado en examen."
                        )
                    else:
                        st.warning(
                            f"Estudiante con ID **{estudiante_id}** del examen no encontrado en la base de datos local."
                        )
                else:
                    st.warning(
                        f"Fila {index+1} del examen subido: Faltan 'ID de Estudiante' o 'Puntuación total'."
                    )

            estudiantes_df_local.to_csv("estudiantes.csv", index=False)
            st.success(
                "Niveles de conocimiento actualizados desde el examen subido. Por favor, recarga la aplicación si deseas ver los cambios de inmediato."
            )
            return estudiantes_df_local
        except Exception as e:
            st.error(f"Error al procesar el archivo CSV de examen: {e}")
            st.info(
                "Asegúrate de que el archivo es un CSV válido y que las columnas 'ID de Estudiante' y 'Puntuación total' existen."
            )
    return estudiantes_df_local


# Función para generar embeddings (copiada del script de indexación)
@st.cache_data(show_spinner=False) # Caching para evitar regenerar el mismo embedding si la consulta se repite
def get_embedding(text, task_type="RETRIEVAL_QUERY", retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type=task_type # Usar RETRIEVAL_QUERY para consultas
            )
            return response['embedding']
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"Falló la generación de embedding después de {retries} intentos: {e}")
                return None


# Función principal para obtener contenido con RAG
@st.cache_data(show_spinner=False)
def obtener_contenido_gemini(estudiante_id, concepto_id, nivel_dificultad_texto):
    nombre_concepto = conceptos_legibles.get(concepto_id, concepto_id)
    # Define la "pregunta" que usaremos para buscar en Pinecone
    query_for_rag = f"Explica en detalle {nombre_concepto} para un nivel {nivel_dificultad_texto} en el contexto de Flipped Classroom."

    # 1. Generar el embedding de la pregunta
    query_embedding = get_embedding(query_for_rag, task_type="RETRIEVAL_QUERY")
    if query_embedding is None:
        return f"No se pudo generar el embedding para buscar información sobre {nombre_concepto}."

    # 2. Buscar en Pinecone para recuperar los fragmentos más relevantes
    context_chunks = []
    relevant_citations = set() # Usar un set para almacenar tuplas (cite_apa, reference_apa) y evitar duplicados

    with st.spinner(f"Buscando información relevante en la base de conocimiento para '{nombre_concepto}'..."):
        try:
            # top_k determina cuántos fragmentos recuperar. Ajusta según sea necesario.
            query_results = index.query(
                vector=query_embedding,
                top_k=5, # Recupera los 5 fragmentos más similares
                include_metadata=True # Asegúrate de obtener los metadatos que guardaste
            )

            for match in query_results.matches:
                # Recupera el texto original del fragmento y los metadatos de cita
                # ¡IMPORTANTE!: Asegúrate de que el script de indexación guarde 'original_text'
                # y los campos de cita en los metadatos del vector.
                chunk_text = match.metadata.get("original_text", "")
                source_filename = match.metadata.get("source_filename", "Desconocido")

                if chunk_text:
                    context_chunks.append(chunk_text)

                # Busca la información de cita usando el nombre del archivo fuente
                # Fallback a los metadatos del match si el diccionario no tiene el archivo
                cite_apa = citations_dict.get(source_filename, {}).get("cite_apa", match.metadata.get("cite_apa", "N/A"))
                reference_apa = citations_dict.get(source_filename, {}).get("reference_apa", match.metadata.get("reference_apa", "N/A"))

                if cite_apa != "N/A" and reference_apa != "N/A":
                    relevant_citations.add((cite_apa, reference_apa))

        except Exception as e:
            st.error(f"Error al consultar Pinecone: {e}. Revisa la conexión y el índice.")
            return f"No se pudo recuperar información de la base de conocimiento para **{nombre_concepto}**."

    # 3. Construir el prompt para Gemini con el contexto recuperado
    context_string = "\n\n".join(context_chunks) if context_chunks else "No se encontró información relevante en los documentos."

    # Prompt detallado para el modelo Gemini, incluyendo el contexto RAG
    rag_prompt = f"""
    Eres un tutor educativo experto en la metodología Flipped Classroom.
    El estudiante con ID {estudiante_id} requiere contenido sobre **{nombre_concepto}**.
    El nivel de conocimiento del estudiante es **{nivel_dificultad_texto}**.

    **INFORMACIÓN DE CONTEXTO RELEVANTE (de tus documentos):**
    {context_string}

    **INSTRUCCIONES PARA GENERAR EL CONTENIDO:**
    1.  Explica el concepto de manera clara y concisa, **utilizando y refiriéndote a la "INFORMACIÓN DE CONTEXTO RELEVANTE" siempre que sea posible.**
    2.  Adapta la complejidad del lenguaje al nivel **{nivel_dificultad_texto}** del estudiante.
    3.  Incluye al menos dos ejemplos prácticos o escenarios relevantes relacionados con Flipped Classroom.
    4.  Usa un tono informativo, pedagógico y motivador.
    5.  Formatea la respuesta en Markdown con encabezados (##, ###), listas y negritas para mejorar la legibilidad.
    6.  **Si utilizaste información del contexto, cita las fuentes directamente en el texto** usando el formato (Autor, Año) o (Autor1 y Autor2, Año).
    7.  Al final de tu explicación, añade una sección de "Referencias Bibliográficas" con las referencias APA completas de las fuentes citadas.

    **Ejemplo de formato de respuesta:**
    ## La Definición de Flipped Classroom (Nivel: Intermedio)

    El concepto de Flipped Classroom, o aula invertida, ... (según Smith, 2020).

    ### Ejemplos Prácticos
    1.  Un ejemplo ...
    2.  Otro escenario ...

    ---
    **Referencias Bibliográficas:**
    - Smith, J. D. (2020). Título del Libro o Artículo. Editorial/Revista.
    """

    try:
        with st.spinner(f"Generando contenido con IA avanzada (RAG) para '{nombre_concepto}'..."):
            response = model.generate_content(rag_prompt)
            generated_content = response.text

            # Añadir las referencias APA al final de la respuesta si hay alguna
            if relevant_citations:
                generated_content += "\n\n---\n**Referencias Bibliográficas:**\n"
                # Ordenar las referencias para una presentación limpia
                sorted_citations = sorted(list(relevant_citations), key=lambda x: x[0])
                for cite_apa, ref_apa in sorted_citations:
                    generated_content += f"- {ref_apa}\n"
            return generated_content
    except Exception as e:
        st.error(f"Error al generar contenido con Gemini (RAG) para '{nombre_concepto}': {e}. Esto podría deberse a un problema con la API o al contenido.")
        return f"No se pudo generar contenido dinámico basado en documentos para **{nombre_concepto}**. Por favor, inténtalo de nuevo más tarde."


# --- Interfaz Streamlit ---

st.set_page_config(layout="wide", page_title="Sistema de Aprendizaje Personalizado")
st.title("Sistema de Aprendizaje Personalizado - Flipped Classroom")

st.subheader("1. Actualizar Niveles con Resultados de Examen (CSV)")
st.info(
    "Descarga el CSV de respuestas de tu Google Form y súbelo aquí para actualizar los niveles de conocimiento de los estudiantes."
)
uploaded_file = st.file_uploader("Sube el archivo CSV de respuestas del examen", type=["csv"])
if uploaded_file is not None:
    # Asegúrate de que estudiantes_df se actualice globalmente o se maneje con st.session_state
    estudiantes_df = actualizar_niveles_desde_csv_examen(estudiantes_df.copy(), uploaded_file)
    # Considera una recarga suave o uso de session_state para reflejar los cambios inmediatamente
    # st.experimental_rerun() # Descomentar si quieres un recarga forzada

st.subheader("2. Consulta tu Contenido Personalizado")
estudiante_id = st.number_input("Ingresa tu ID de estudiante", min_value=1, step=1, key="student_id_input")

if estudiante_id:
    estudiante = estudiantes_df[estudiantes_df["id"] == estudiante_id]
    if not estudiante.empty:
        st.write(f"**Nombre del estudiante**: {estudiante['nombre'].iloc[0]}")

        st.subheader("Tu Nivel de Conocimiento Actual:")
        niveles_para_grafico = []
        for concepto in conceptos:
            nivel_num = estudiante[concepto].iloc[0]
            nivel_texto = nivel_map.get(nivel_num, "Desconocido")
            niveles_para_grafico.append(
                {"Concepto": concepto, "Nivel": nivel_texto, "Valor": nivel_num}
            )

        niveles_df_chart = pd.DataFrame(niveles_para_grafico)

        chart = (
            alt.Chart(niveles_df_chart)
            .mark_bar()
            .encode(
                x=alt.X("Concepto:N", title="Concepto", sort=conceptos),
                y=alt.Y("Valor:Q", title="Nivel de Conocimiento", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Nivel:N",
                    scale=alt.Scale(
                        domain=["Básico", "Intermedio", "Avanzado"],
                        range=["#ff9999", "#66b3ff", "#99ff99"],
                    ),
                ),
                tooltip=["Concepto", "Nivel", "Valor"],
            )
            .properties(width="container", height=300, title="Niveles de Conocimiento por Concepto")
            .configure_title(dy=-10)
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown("---")

        st.subheader("Contenido Personalizado para Ti:")
        for concepto in conceptos:
            nivel_num = estudiante[concepto].iloc[0]
            nivel_dificultad_texto = nivel_map.get(nivel_num, "Básico")
            with st.expander(f"Contenido para {conceptos_legibles.get(concepto, concepto).capitalize()} (Nivel: {nivel_dificultad_texto})"):
                st.markdown(obtener_contenido_gemini(estudiante_id, concepto, nivel_dificultad_texto))
    else:
        st.error("ID de estudiante no encontrado. Por favor, verifica el ID.")

st.subheader("3. Actualizar Nivel de Conocimiento Manualmente (Administrador)")
with st.form("actualizar_conocimiento_manual"):
    estudiante_id_update = st.number_input(
        "ID del estudiante a actualizar", min_value=1, step=1, key="manual_update_id"
    )
    concepto_update = st.selectbox("Concepto a actualizar", conceptos, key="manual_update_concept")
    nuevo_nivel = st.selectbox(
        "Nuevo nivel", [0.25, 0.55, 0.85], format_func=lambda x: nivel_map[x], key="manual_update_level"
    )
    submit = st.form_submit_button("Actualizar Nivel")

    if submit:
        idx = estudiantes_df[estudiantes_df["id"] == estudiante_id_update].index
        if not idx.empty:
            estudiantes_df.loc[idx, concepto_update] = nuevo_nivel
            estudiantes_df.to_csv("estudiantes.csv", index=False)
            st.success(
                f"Nivel de conocimiento para **{concepto_update}** actualizado a **{nivel_map[nuevo_nivel]}** para el estudiante ID {estudiante_id_update}."
            )
            # st.experimental_rerun() # Descomentar si quieres recarga forzada para ver el gráfico actualizado
        else:
            st.error("ID de estudiante no encontrado. No se pudo actualizar el nivel.")
