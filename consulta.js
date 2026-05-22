import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/ollama";

async function buscarInformacion() {
    console.log("1. Conectando al modelo de Embeddings...");
    const embeddings = new OllamaEmbeddings({
        model: "nomic-embed-text", 
        maxConcurrency: 5,
    });

    console.log("2. Conectando a tu ChromaDB local...");
    const vectorStore = new Chroma(embeddings, {
        collectionName: "documentos_rag",
        url: "http://localhost:8000", 
    });

    
    const pregunta = "¿de que trata este documento?"; 
    
    console.log(`\nBuscando respuesta a: "${pregunta}"...\n`);

    const resultados = await vectorStore.similaritySearch(pregunta, 2);

    console.log("¡Resultados encontrados!");
    resultados.forEach((res, i) => {
        console.log(`\n--- Fragmento Recuperado ${i + 1} ---`);
        console.log(res.pageContent);
    });
}

buscarInformacion().catch(console.error);