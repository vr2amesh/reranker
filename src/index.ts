import { Context, Layer, Effect, Runtime, Console } from "effect"
import express from "express";
import { MixedbreadAIClient } from "@mixedbread-ai/sdk";
import { similarity } from "ml-distance";
import fs from 'fs';
import path from 'path';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAI } from 'openai';
import model from 'wink-eng-lite-web-model';

type Message = {
    content: string;
    role: string;
}

class Express extends Context.Tag("Express")<
  Express,
  ReturnType<typeof express>
>() {}

const mxbai = new MixedbreadAIClient({ apiKey: 'emb_71bc347f9db377fc9ce3077e88a56a3a9d58dc6e64f87a66' });
const embedModelName = "mixedbread-ai/mxbai-embed-large-v1";
const rerankModelName = "mixedbread-ai/mxbai-rerank-large-v1";
const pc = new Pinecone({
    apiKey: '8b411b5c-4e68-4df1-91f4-ec5d54470790'
});
const openai = new OpenAI({
    apiKey: 'sk-proj-dhmaQTbdJQ3AIn2IiZfxT3BlbkFJLx7FFZKdDn2TuPijHPbF'
});
const nlp = require('wink-nlp')(model);
const its = nlp.its;
const BM25Vectorizer = require('wink-nlp/utilities/bm25-vectorizer');
const bm25 = BM25Vectorizer();

const supportDocs = ['google-support.txt', 'twitter-support.txt', 'github-support.txt', 'discord-support.txt']

async function summarizeStrings(messages: Message[]): Promise<string | undefined> {
    try {
      // Join the list of strings into a single prompt
      const prompt = `You are given a series of questions that have been asked by a user looking to get answers. Please take the series of questions and summarize the issue they want to ask to a chat assistant, so that the chat assistant may easily understand the question:\n\n${messages.map(m=>m.content).join('\n\n')}`;
  
      // Send the prompt to the OpenAI API
      const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: prompt },
        ],
      });
  
      // Extract and return the summary from the response
      const summary = response.choices[0].message.content;
      return summary ?? '';
    } catch (error) {
      console.error('Error summarizing strings:', error);
    }
}

const readDocument = async (filePath: string): Promise<string[]> => {
    return new Promise((resolve, reject) => {
        fs.readFile(path.join(__dirname, 'documents', filePath), 'utf8', (err, data) => {
            if (err) {
                reject(err);
            } else {
                // Split the file content into paragraphs
                const paragraphs = data.split('\n\n').filter(paragraph => paragraph.trim() !== '');
                resolve(paragraphs);
            }
        });
    });
};

const getEmbeddings = async (
    input: string[],
    model: string,
    prompt?: string,
) => {
    const res = await mxbai.embeddings({
        input,
        model,
        prompt,
    });
    return res.data.map((entry) => entry.embedding);
};

const getTopSimilarDocuments = async (messages: Message[], documentPaths: string[], embedModelName: string, topN: number = 1) => {
    const queryEmbeddings = await getEmbeddings(messages.map(m => m.content), embedModelName);

    const documentEmbeddingsPromises = documentPaths.map(async (documentPath) => {
        const documentParagraphs = await readDocument(documentPath);
        return getEmbeddings(documentParagraphs, embedModelName);
    });
    const documentEmbeddings = await Promise.all(documentEmbeddingsPromises);

    const flatDocumentEmbeddings = documentEmbeddings.flat();

    const similarities = queryEmbeddings.map(queryEmbedding => {
        return flatDocumentEmbeddings.map(docEmbedding => similarity.cosine(queryEmbedding, docEmbedding));
    });

    const averageSimilarities = documentEmbeddings.map((embeddings, docIdx) => {
        const docSimilarities = embeddings.map((_, paraIdx) => similarities.map(sim => sim[paraIdx + docIdx * embeddings.length]).reduce((sum, val) => sum + val, 0) / queryEmbeddings.length);
        return docSimilarities.reduce((sum, val) => sum + val, 0) / docSimilarities.length;
    });

    const topNIndices = averageSimilarities
        .map((sim, idx) => ({ sim, idx }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, topN)
        .map(entry => entry.idx);

    return topNIndices.map(idx => documentPaths[idx]);
};

const generateResponse = async (topSimilarDocumentsContent: string[], summary: string) => {
    try {
        // Join the list of strings into a single prompt
        const prompt = `You are responding to a user's request for customer support. Here is their question:\n
            ${summary}\n

            Based on this question, I have curated some documents that might be useful for you to answer this user, here are 
            those documents in order of potential relevance to the answer. Please use this information to generate a response.\n

            1. ${topSimilarDocumentsContent[0]}\n

            2. ${topSimilarDocumentsContent[1]}\n

            3. ${topSimilarDocumentsContent[2]}\n

            4. ${topSimilarDocumentsContent[3]}\n
        `;
    
        // Send the prompt to the OpenAI API
        const response = await openai.chat.completions.create({
            model: 'gpt-4',
            messages: [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: prompt },
            ],
        });
    
        // Extract and return the summary from the response
        const responseToUser = response.choices[0].message.content;
        return responseToUser ?? '';
    } catch (error) {
        console.error('Error generating response:', error);
    }
}


const ChatRouteLive = Layer.effectDiscard(
    Effect.gen(function* () {
        const app = yield* Express
        const runFork = Runtime.runFork(yield* Effect.runtime<never>())

        app.use(express.json());

        app.get("/chat", (req, res) => {
            runFork(Effect.sync(async () => {
                // 1. Convert the input query into a condensed one that summarizes all the text
                const { messages } = req.body as { messages: Message[] };
                const summary = await summarizeStrings(messages);
                // 2. Get dense vectors (mixedbread)

                // 3. Get sparse vectors (wink)
                bm25.learn(nlp.readDoc(await readDocument('google-support.txt')).tokens().out(its.normal));
                const sparseQueryEmbeddings = bm25.vectorOf(nlp.readDoc(summary).tokens().out(its.normal));

                const topSimilarDocuments = await getTopSimilarDocuments(messages, supportDocs, embedModelName, 4);
                console.log(topSimilarDocuments);

                const topSimilarDocumentsContent = await Promise.all(topSimilarDocuments.map(async doc => (await readDocument(doc)).join('\n\n')));
                const response = await mxbai.reranking({
                    model: rerankModelName,
                    query: summary as string,
                    input: topSimilarDocumentsContent,
                    topK: 4, 
                    returnInput: false
                });
                const newDocuments = response.data.map(doc => topSimilarDocumentsContent[doc.index])
                console.log(response.data.map(doc => topSimilarDocuments[doc.index]))
                
                const chatResponse = await generateResponse(newDocuments, summary as string);
                // 4. Create index with dotproduct metric
                // const now = Date.now();
                // const index = await pc.createIndex({
                //     name: `severless-index`,
                //     dimension: 1536,
                //     metric: 'dotproduct',
                //     spec: {
                //         serverless: {
                //             cloud: 'aws',
                //             region: 'us-east-1'
                //         }
                //     },
                //     waitUntilReady: true
                // });
                // 5. Upsert sparse-dense vectors to index
                // await pc.index(`serverless-index`).upsert([{
                //     id: 'vec1',
                //     values: denseQueryEmbeddings[0] as number[],
                //     sparseValues: sparseQueryEmbeddings,
                // }]);
                // 6. Search the index using sparse-dense vectors
                // 7. Pinecone returns sparse-dense vectors

                res.json({
                    chatResponse
                })
            }))
        })
    })
)


const ServerLive = Layer.scopedDiscard(
    Effect.gen(function* () {
        const port = 3001
        const app = yield* Express
        yield* Effect.acquireRelease(
            Effect.sync(() =>
                app.listen(port, () =>
                    console.log(`Example app listening on port ${port}`)
                )
            ),
            (server) => Effect.sync(() => server.close())
        )
    })
)


const ExpressLive = Layer.sync(Express, () => express())


const AppLive = ServerLive.pipe(
    Layer.provide(ChatRouteLive),
    Layer.provide(ExpressLive)
)


Effect.runFork(Layer.launch(AppLive))
