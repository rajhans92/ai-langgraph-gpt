import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { StateGraph, START, END } from "@langchain/langgraph";
import { SystemMessage } from "@langchain/core/messages";
import { MessagesZodMeta } from "@langchain/langgraph";
import { isAIMessage } from "@langchain/core/messages";
import { registry } from "@langchain/langgraph/zod";
import { HumanMessage } from "@langchain/core/messages";
import * as z from "zod";

const MessagesState = z.object({
    messages: z
      .array(z.custom())
      .register(registry, MessagesZodMeta),
    llmCalls: z.number().optional(),
  });

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

// Define tools
const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

// Augment the LLM with tools
const toolsByName = {
  [add.name]: add,
  [multiply.name]: multiply,
  [divide.name]: divide,
};
const tools = Object.values(toolsByName);
const modelWithTools = model.bindTools(tools);

async function llmCall(state) {
    return {
      messages: await modelWithTools.invoke([
        new SystemMessage(
          "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        ),
        ...state.messages,
      ]),
      llmCalls: (state.llmCalls ?? 0) + 1,
    };
  }

  async function toolNode(state) {
    const lastMessage = state.messages.at(-1);
  
    if (lastMessage == null || !isAIMessage(lastMessage)) {
      return { messages: [] };
    }
  
    const result = [];
    for (const toolCall of lastMessage.tool_calls ?? []) {
      const tool = toolsByName[toolCall.name];
      const observation = await tool.invoke(toolCall);
      result.push(observation);
    }
  
    return { messages: result };
  }
  
  async function shouldContinue(state) {
    const lastMessage = state.messages.at(-1);
    if (lastMessage == null || !isAIMessage(lastMessage)) return END;
  
    // If the LLM makes a tool call, then perform an action
    if (lastMessage.tool_calls?.length) {
      return "toolNode";
    }
  
    // Otherwise, we stop (reply to the user)
    return END;
  }

const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile();

// Invoke
const result = await agent.invoke({
  messages: [new HumanMessage("Add 3 and 4.")],
});

for (const message of result.messages) {
  console.log(`[${message.getType()}]: ${message.text}`);
}
