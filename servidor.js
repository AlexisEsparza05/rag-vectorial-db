import express from 'express';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings, Ollama } from "@langchain/ollama";

const app = express();
// Permitimos que el servidor entienda datos en formato JSON
app.use(express.json()); 

// 1. Inicializamos las conexiones a la IA y la Base de Datos
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text", maxConcurrency: 5 });
const llm = new Ollama({ model: "llama3", temperature: 0 });
const vectorStore = new Chroma(embeddings, {
    collectionName: "documentos_rag",
    url: "http://localhost:8000"
});

// 2. Creamos la ruta de la API (Endpoint)
app.post('/api/preguntar', async (req, res) => {
    try {
        const { pregunta } = req.body;
        
        if (!pregunta) {
            return res.status(400).json({ error: "Por favor, envía una pregunta." });
        }

        console.log(`\nRecibiendo pregunta: "${pregunta}"`);

        // Recuperación (Retrieval)
        const resultados = await vectorStore.similaritySearch(pregunta, 2);
        const contextoCrudo = resultados.map(res => res.pageContent).join("\n\n");

        // Generación (Generation)
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

        const respuestaFinal = await llm.invoke(promptStrict);
        
        // 3. Enviamos la respuesta de vuelta al usuario
        res.json({
            respuesta: respuestaFinal,
            fuentes_consultadas: resultados.length
        });

    } catch (error) {
        console.error("Error en el servidor:", error);
        res.status(500).json({ error: "Ocurrió un error al procesar tu consulta." });
    }
});

// Encendemos el servidor
const PUERTO = 3000;
app.listen(PUERTO, () => {
    console.log(`🚀 Servidor RAG funcionando en http://localhost:${PUERTO}`);
    console.log(`Esperando preguntas en el endpoint POST /api/preguntar...`);
});