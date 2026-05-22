import { Ollama } from "@langchain/ollama";

async function evaluarSistemaCompleto() {
    console.log("Iniciando prueba de QA: Evaluación completa del Sistema RAG...");

    const pregunta = "¿Quiénes estuvieron involucrados en el proyecto de Darkys B2B Hub?";
    const contextoRecuperado = "Todo el equipo de desarrollo (Gerardo Esparza, Johnny Ayala, Andrés Santiago) y el dueño del negocio Roberto Leal. Se cerró la fase de cimentación...";
    const respuestaGenerada = "Los involucrados en el proyecto Darkys B2B Hub fueron el equipo de desarrollo: Gerardo Esparza, Johnny Ayala, y Andrés Santiago, además del dueño del negocio, Roberto Leal.";

    // --- FIX: Obligamos a Llama 3 a responder en JSON estricto ---
    const juezLlm = new Ollama({ 
        model: "llama3", 
        temperature: 0,
        format: "json" // Esta es la configuración mágica
    });

    const promptEvaluacion = `
    Eres un juez experto evaluando el rendimiento de un sistema RAG (Retrieval-Augmented Generation). 
    Debes analizar la siguiente interacción y otorgar una calificación del 1 al 10 para cuatro métricas específicas.

    INTERACCIÓN A EVALUAR:
    - Pregunta del usuario: "${pregunta}"
    - Contexto recuperado de la base de datos: "${contextoRecuperado}"
    - Respuesta final del sistema: "${respuestaGenerada}"

    MÉTRICAS A EVALUAR:
    1. Precisión del Contexto
    2. Recuerdo del Contexto
    3. Fidelidad
    4. Relevancia de la Respuesta

    Responde ÚNICAMENTE con un objeto JSON válido. NO uses comillas dobles dentro de las justificaciones (usa comillas simples si es necesario).
    Sigue esta estructura exacta:
    {
      "precision_contexto": {"puntuacion": 0, "justificacion": "texto explicativo corto"},
      "recuerdo_contexto": {"puntuacion": 0, "justificacion": "texto explicativo corto"},
      "fidelidad": {"puntuacion": 0, "justificacion": "texto explicativo corto"},
      "relevancia_respuesta": {"puntuacion": 0, "justificacion": "texto explicativo corto"}
    }
    `;

    console.log("Analizando métricas (Precisión, Recuerdo, Fidelidad y Relevancia)...\n");
    
    try {
        const resultadoStr = await juezLlm.invoke(promptEvaluacion);
        
        // El salvavidas: Intentamos convertirlo a JSON colorido, si falla, mostramos el texto crudo
        try {
            const resultadoJSON = JSON.parse(resultadoStr);
            console.log("===========================================");
            console.log("📊 REPORTE FINAL DE MÉTRICAS RAG:");
            console.log("===========================================");
            console.dir(resultadoJSON, { depth: null, colors: true });
        } catch (parseError) {
            console.log("===========================================");
            console.log("📊 REPORTE (Formato Crudo):");
            console.log("===========================================");
            console.log(resultadoStr);
        }

    } catch (error) {
        console.error("Error al evaluar (Asegúrate de que Ollama esté corriendo):", error);
    }
}

evaluarSistemaCompleto().catch(console.error);