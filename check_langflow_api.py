import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if available)
load_dotenv()

# API configuration
LANGFLOW_URL = "http://192.168.112.128:7860/api/v1/run/nasa-dataset-selector-02-1-1"
LANGFLOW_API_KEY = "sk-HCJT-1SVbp5BvqI4EL27OK3CpVzeWpmXhBI-CkjlQ9k"

# Test query
test_query = "I need data for precipitation in RCP8.5 scenario from 2040 to 2060"


def check_api():
    print(f"Testing Langflow API with query: {test_query}")

    # Prepare the API request payload
    payload = {
        "session_id": "test-session",
        "output_type": "debug",
        "input_type": "chat",
        "input_value": test_query,
    }

    headers = {"Content-Type": "application/json", "x-api-key": LANGFLOW_API_KEY}

    try:
        # Make the API request
        print("Sending request to API...")
        response = requests.post(
            LANGFLOW_URL,
            headers=headers,
            json=payload,
            params={"stream": "false"},
            timeout=180,
        )

        # Check response status
        response.raise_for_status()
        print(f"API response status: {response.status_code}")

        # Parse the response
        result = response.json()

        # Print some basic information from the response
        print("\nResponse received successfully!")
        print(f"Response size: {len(json.dumps(result))} characters")

        # Check if outputs exist in the response
        if "outputs" in result and result["outputs"]:
            print("API returned outputs as expected.")

            # Optionally save the response to a file
            with open("api_test_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print("Full response saved to api_test_response.json")
        else:
            print("Warning: API response doesn't contain expected 'outputs' field.")

        return True

    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to the API. Please check the URL and network connection."
        )
    except requests.exceptions.Timeout:
        print(
            "Error: Request timed out. The API server might be overloaded or unreachable."
        )
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(
            f"Response text: {e.response.text if hasattr(e, 'response') else 'No response text'}"
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    return False


if __name__ == "__main__":
    success = check_api()
    if success:
        print("\n✅ Langflow API is working!")
    else:
        print("\n❌ Failed to connect to Langflow API.")
