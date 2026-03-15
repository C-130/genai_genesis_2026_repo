import webbrowser
import urllib.parse
from moorcheh_sdk import MoorchehClient
from moorcheh_sdk.exceptions import ConflictError
import os
import re
import uuid
from dotenv import load_dotenv
import json

load_dotenv()

HEADER_PROMPT = "You are an assistant that helps users compose emails based on natural language requests."

PROMPT = (
    "The following is a natural language request to compose an email. "
    "Extract the recipient's email address and CC email address (if any). Also extract the subject and body of the email if explicitly specified, and come up with an appropriate subject and body if not specified."
    "Return the information in a json object with keys: 'address', 'recipient', 'stored', 'subject', 'cc', and 'body'. "
    "If the request mentions a recipient has been stored in the namespace, use that email address, return the recipient name in the attribute 'recipient', and return True in the attribute 'stored' in the result."
    "Otherwise, return False in the 'stored' attribute, extract the email address from the request, and come up with a suitable recipient name and store it in the attribute 'recipient'. "
    "If any of the fields are not specified in the request, return an empty string for that field.\n\nRequest: {query}"
)

FOOTER_PROMPT = "Return only the json object without any additional text or explanation."

MOORCHEH_NAMESPACE = "user_email_addresses"


def open_email_draft(address: str, subject: str, cc: str, body: str):
    """
    Opens the user's default email client with a pre-filled draft.

    Parameters
    ----------
    address : str
        Primary recipient email address
    subject : str
        Email subject line
    cc : str
        CC email address (can be empty string)
    body : str
        Email body text
    """

    params = {}

    if subject:
        params["subject"] = urllib.parse.quote(subject)

    if cc:
        params["cc"] = urllib.parse.quote(cc)

    if body:
        params["body"] = urllib.parse.quote(body)
    param_list = [f"{key}={value}" for key, value in params.items() if len(value) > 0]
    param_component = "&".join(param_list)

    mailto_url = f"mailto:{address}?{param_component}"

    webbrowser.open(mailto_url)


def interpret_email_request(query: str) -> dict:
    """
    Uses Moorcheh to interpret a natural language email request.

    Parameters
    ----------
    query : str
        User's natural language request for composing an email

    Returns
    -------
    dict
        Dictionary containing 'address', 'subject', 'cc', and 'body' keys
    """
    print(os.getenv("MOORCHEH_API_KEY"))
    with MoorchehClient(api_key=os.getenv("MOORCHEH_API_KEY")) as client:
        try:
            client.create_namespace(namespace_name=MOORCHEH_NAMESPACE, type="text")
        except ConflictError:
            print("Namespace already exists.\n")
        response = client.answer.generate(
            namespace=MOORCHEH_NAMESPACE,
            query=PROMPT.format(query=query),
            temperature=0.5,
            header_prompt=HEADER_PROMPT,
            footer_prompt=FOOTER_PROMPT,
        )
        answer = response["answer"].strip()
        print(answer)
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        data = None
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            print(data)
        assert {"address", "recipient", "stored", "subject", "cc", "body"}.issubset(data.keys()), "invalid response."
        if not data["stored"]:
            documents_to_upload = [
                {
                    "id": str(uuid.uuid4()),
                    "text": str({"address": data["address"], "recipient": data["recipient"]})
                }
            ]
            client.documents.upload(MOORCHEH_NAMESPACE, documents_to_upload)
        return {"address": data["address"], "subject": data["subject"], "cc": data["cc"], "body": data["body"]}


if __name__ == "__main__":
    query = "Send an email to Rumplestiltskin to tell him that I'll go to Dresden on 2026/05/23 for vacation. Ask him if he can pick me up at the local train station after I arrived via the iconic ICE3 train, and help me think of a natural subject for this email. Cc this email to outreach@dbrail.de"
    email_data = interpret_email_request(query)
    print(email_data)
    if email_data["address"]:
        open_email_draft(address=email_data["address"], subject=email_data["subject"], cc=email_data["cc"], body=email_data["body"])
    else:
        print("No email address found in the request.")