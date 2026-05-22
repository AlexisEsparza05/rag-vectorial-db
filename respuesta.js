import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama"; // Importamos el generador de texto

async function generarRespuesta() {
    console.log("1. Conectando al motor local...");
    
    // El modelo matemático para buscar
    const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text", maxConcurrency: 5 });
    
    // El modelo de lenguaje que Jhonny configuró (con temperatura 0 para que no invente)
    const llm = new Ollama({ model: "llama3", temperature: 0 }); 

    const vectorStore = new Chroma(embeddings, {
        collectionName: "documentos_rag",
        url: "http://localhost:8000", 
    });

    const pregunta = "¿De qué trata el proyecto Darkys B2B Hub y quiénes estuvieron involucrados?"; 
    
    console.log("2. Recuperando información de ChromaDB...");
    const resultados = await vectorStore.similaritySearch(pregunta, 2);

    // Unimos los fragmentos crudos en un solo bloque de texto
    const contextoCrudo = resultados.map(res => res.pageContent).join("\n\n");

    console.log("3. Pasando contexto a la Inteligencia Artificial (Lógica Cognitiva)...");

    // AQUÍ ESTÁ LA CLAVE PARA EL SOBRESALIENTE: El System Prompt Restrictivo
    const promptStrict = `
    Eres un asistente corporativo experto y muy estricto. Tu única tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada en el siguiente "Contexto de la Base de Datos".
    
    Regla 1: Si la respuesta a la pregunta no se encuentra explícitamente en el contexto, tienes prohibido inventar o deducir información. Debes responder exactamente con esta frase: "El dato no está en la base de datos Vectorial".
    Regla 2: Tu respuesta debe ser natural, conversacional y resumir claramente los puntos clave.
    Regla 3: Responde siempre en idioma Español.

    --- Contexto de la Base de Datos ---
    ${contextoCrudo}
    ------------------------------------

    Pregunta del usuario: ${pregunta}
    
    Tu respuesta estructurada:
    `;

    // Le enviamos la instrucción al modelo
    const respuestaFinal = await llm.invoke(promptStrict);
    
    console.log("\n===========================================");
    console.log("🤖 RESPUESTA FINAL DEL SISTEMA RAG:");
    console.log("===========================================\n");
    console.log(respuestaFinal);
}

generarRespuesta().catch(console.error);