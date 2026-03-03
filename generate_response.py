import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
from dotenv import load_dotenv, find_dotenv
from agents import GuardrailFunctionOutput, InputGuardrailTripwireTriggered, input_guardrail, RunContextWrapper, TResponseInputItem, output_guardrail
from pydantic import BaseModel
import json
load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY1"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)

class InputGuardrailTrueFalse(BaseModel):
    is_a_job_description: bool
    client_name: str
    has_portfolio_link: bool
    
class OutputGuardrailCheck(BaseModel):
    is_a_proposal: bool
    has_portfolio_link: bool
    
input_guardrail_agent = Agent(
    name="Input Guardrail Check",
    instructions="""
You are a guardrail AI that analyzes text.

Return:
- is_a_job_description: true if this is a job description
- client_name: name of the client if present, otherwise empty string
- has_portfolio_link: true if a portfolio link of the freelancer is included

Respond strictly in the required JSON format.
    """,
    model=llm_model,
    output_type=InputGuardrailTrueFalse
)

output_guardrail_agent = Agent(
    name="Output Guardrail Check",
    instructions="""
You are a guardrail AI that checks if the generated response is a freelancer proposal.

Return:
- is_a_proposal: true if the response is a proper proposal
- has_portfolio_link: true if the proposal includes a valid portfolio link

Respond strictly in JSON.
    """,
    model=llm_model,
    output_type=OutputGuardrailCheck
)

@input_guardrail
async def relevant_detector_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    prompt: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input=prompt)
    is_a_job = result.final_output.is_a_job_description
    has_portfolio = result.final_output.has_portfolio_link
    if not is_a_job:
        return GuardrailFunctionOutput(
        output_info=json.dumps({
        "error" : "TripwireTrigerredError",
        "message" : "Given Input is not a job description."
        }),
        tripwire_triggered=True,
        )
    if not has_portfolio:
        return GuardrailFunctionOutput(
        output_info=json.dumps({
        "error" : "TripwireTrigerredError",
        "message" : "Given Input has not a valid portfolio."
        }),
        tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(
        output_info=json.dumps({
            "status": "success",
            "message": "Valid Job description with portfolio link"
        }),
        tripwire_triggered=False,
    )
    
@output_guardrail
async def ouput_detector_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    response: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, input=response)
    is_proposal = result.final_output.is_a_proposal
    has_portfolio = result.final_output.has_portfolio_link

    if not is_proposal:
        return GuardrailFunctionOutput(
            output_info=json.dumps({
                "error": "TripwireTriggeredError",
                "message": "Generated response is not a proposal."
            }),
            tripwire_triggered=True,
        )

    if not has_portfolio:
        return GuardrailFunctionOutput(
            output_info=json.dumps({
                "error": "TripwireTriggeredError",
                "message": "Proposal does not include portfolio link."
            }),
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info=json.dumps({
            "status": "success",
            "message": "Valid proposal with portfolio link."
        }),
        tripwire_triggered=False,
    )


proposal_agent=Agent(
    name="Assistant",
    instructions=
    """
You are an expert freelance job proposal writer.

You must generate a professional proposal using the exact structure below:

1. Greeting  
   Example: Hello [Client Name],

2. Restate Employer's Project  
   Briefly summarize what the client is looking for.

3. Introduce Yourself  
   Mention your experience, expertise, and credibility.

4. What You Can Do  
   Clearly explain how you will solve the client's problem.
   Mention timeline, revisions, and deliverables.

5. Portfolio  
   Include a real-looking portfolio link (must start with https://).

6. Free Mockup / Question  
   Offer a free mockup or ask a relevant project-related question.

7. Closing  
   End professionally and express interest in working together.

IMPORTANT RULES:
- The proposal MUST include a portfolio link.
- If client name is given then write in proposal, otherwise don't write
- The response MUST be a proper freelancer proposal.
- Return strictly valid JSON.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include text outside JSON.

Return format:

{
  "proposal": "Full proposal text here..."
}
    """,
    model=llm_model,
    input_guardrails=[relevant_detector_guardrail],
    output_guardrails=[ouput_detector_guardrail]
)
