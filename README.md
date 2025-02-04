Live demo at 

https://climate-pal.namastex.ai/flow/e5a6ccad-1aa6-458b-abcc-ed9b9809b5ff/folder/54aad242-ee9a-4a53-960f-3fe4f6177497

Accounts with:

OpenAI (for an API key)
Groq (Free tier for API key)

Supabase (for a database)

1. Install Langflow:

Follow the official installation instructions: https://github.com/langflow-ai/langflow

Recommended installation: uv pip install langflow

Verify installation: langflow --version

2. Launch Langflow:

Run: langflow

Access Langflow in your browser (usually http://127.0.0.1:7860).

3. Import the Flow:

Download Dataset Retrieval - v1.0.3.json from the flow/ directory.

In Langflow, create a new flow and import the downloaded JSON file.

4. API Key Setup:

For the following components, enter your API keys and Supabase details in the component settings:

OpenAIModel (OpenAI API Key)

GroqModel (Groq API Key)

Supabase Save Configuration (Supabase URL, Supabase Key, Table Name)

5. Create a Supabase Server:

Sign up at https://supabase.com/ and create a new project.

Note your "Project URL" and the "service_role" API key from the Supabase dashboard's API settings.

In the Supabase SQL Editor, run the SQL script from sql/create_feedback_results.sql to create the sentiment_feedback table.

Check the flow Notes for more detailed instructions.

With these steps you can open the playground and talk to the Climate Pal agent.