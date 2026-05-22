import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/ollama";

async function procesarDocumento() {
    console.log("1. Cargando el documento PDF...");
    const loader = new PDFLoader("./manual_prueba.pdf"); 
    const docs = await loader.load();

    console.log("2. Fragmentando el texto (Chunking)...");
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, 
        chunkOverlap: 200, 
    });
    const fragmentos = await splitter.splitDocuments(docs);
    console.log(`Se crearon ${fragmentos.length} fragmentos de texto.`);

    // --- LA SOLUCIÓN: Limpieza de metadatos complejos ---
    // Recorremos cada fragmento y borramos los objetos problemáticos
    fragmentos.forEach(doc => {
        if (doc.metadata && doc.metadata.pdf) {
            delete doc.metadata.pdf;
        }
        if (doc.metadata && doc.metadata.loc) {
            delete doc.metadata.loc; 
        }
    });

    console.log("3. Conectando con el modelo de Embeddings...");
    const embeddings = new OllamaEmbeddings({
        model: "nomic-embed-text", 
        maxConcurrency: 5,
    });

    console.log("4. Guardando en ChromaDB...");
    const vectorStore = await Chroma.fromDocuments(fragmentos, embeddings, {
        collectionName: "documentos_rag",
        url: "http://localhost:8000", 
    });

    console.log("¡Ingesta completada con éxito! La base de datos vectorial está lista.");
}

procesarDocumento().catch(console.error);