import requests
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class SuccessResponseDataFormat:
    total_llm_calls: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    final_llm_response: Dict[str, Any]
    llm_responses_arr: List[Dict[str, Any]]
    messages: List[str]
    output_type: str

@dataclass
class LlmResponseStruct:
    Data: Optional[Dict[str, Any]]
    Error: Optional[Union[Exception, str, Dict[str, Any]]]
    Status: bool

@dataclass
class GeminiChatCompletionParams:
    input: str = ''
    images_arr: List[Any] = field(default_factory=list)
    input_type: str = 'text'
    is_stream: bool = False
    prompt: str = ''
    api_key: str = ''
    chat_model: str = 'gemini-1.5-flash'
    vision_model: str = 'gemini-1.5-pro-vision'
    speech_model: str = ''
    chat_history: List[ChatMessage] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 1000
    forced_tool_calls: Optional[Any] = None
    tool_choice: str = 'auto'

async def gemini_processor(data: Dict[str, Any]) -> LlmResponseStruct:
    """Gemini LLM Processor with enhanced error handling and debugging"""
    # Debug: Show raw input structure
    print("\n=== [1] RAW INPUT ===")
    print(json.dumps(data, indent=2))
    
    try:
        # Validate selected_client first
        if data.get('selected_client') != "MCP_CLIENT_GEMINI":
            error_msg = {
                "message": "Invalid client selection",
                "expected": "MCP_CLIENT_GEMINI",
                "received": data.get('selected_client'),
                "solution": "Set selected_client to 'MCP_CLIENT_GEMINI'"
            }
            return LlmResponseStruct(Data=None, Error=error_msg, Status=False)

        # Extract parameters
        client_details = data.get('client_details', {})
        print("\n=== [2] CLIENT DETAILS ===")
        print(json.dumps(client_details, indent=2))

        # Parse all parameters from client_details
        params = GeminiChatCompletionParams(
            input=client_details.get('input', ''),
            images_arr=client_details.get('images_arr', []),
            input_type=client_details.get('input_type', 'text'),
            is_stream=client_details.get('is_stream', False),
            prompt=client_details.get('prompt', ''),
            api_key=client_details.get('api_key', ''),
            chat_model=client_details.get('chat_model', 'gemini-1.5-flash'),
            vision_model=client_details.get('vision_model', 'gemini-1.5-pro-vision'),
            speech_model=client_details.get('speech_model', ''),
            chat_history=[ChatMessage(**msg) if isinstance(msg, dict) else msg
                         for msg in client_details.get('chat_history', [])],
            tools=client_details.get('tools', []),
            temperature=client_details.get('temperature', 0.1),
            max_tokens=client_details.get('max_tokens', 1000),
            forced_tool_calls=client_details.get('forced_tool_calls'),
            tool_choice=client_details.get('tool_choice', 'auto')
        )
        
        print("\n=== [3] PARSED PARAMS ===")
        print(json.dumps(asdict(params), indent=2))

        # Validate required fields
        if not params.api_key:
            return LlmResponseStruct(
                Data=None,
                Error={
                    "message": "API key required",
                    "field": "api_key",
                    "location": "client_details"
                },
                Status=False
            )

        if not params.prompt and not params.input:
            return LlmResponseStruct(
                Data=None,
                Error={
                    "message": "Either prompt or input must be provided",
                    "fields": ["prompt", "input"],
                    "location": "client_details"
                },
                Status=False
            )

        # Model selection and validation
        selected_model = params.vision_model if params.input_type == 'image' else params.chat_model
        valid_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-pro-vision"]
        
        if selected_model not in valid_models:
            return LlmResponseStruct(
                Data=None,
                Error={
                    "message": "Unsupported model",
                    "received": selected_model,
                    "valid_models": valid_models
                },
                Status=False
            )

        # Build chat contents
        chat_contents = []
        for msg in params.chat_history:
            if msg.role in ['user', 'model']:
                chat_contents.append({
                    "role": msg.role,
                    "parts": [{"text": msg.content}]
                })

        chat_contents.append({
            "role": "user",
            "parts": [{"text": params.input}]
        })

        # Build payload
        payload = {
            "system_instruction": {
                "parts": [{"text": params.prompt}]
            },
            "contents": chat_contents,
            "generationConfig": {
                "temperature": params.temperature,
                "maxOutputTokens": params.max_tokens
            }
        }

        # Handle tools if present
        if params.tools:
            function_declarations = []
            for tool in params.tools:
                func = tool.get("function", {})
                parameters = func.get("parameters", {})
                props = parameters.get("properties", {})

                processed_props = {}
                for key, val in props.items():
                    if val.get("type") == "array":
                        processed_props[key] = {
                            "type": "array",
                            "items": {"type": val.get("items", {}).get("type", "string")},
                            "default": val.get("default", []),
                            "description": val.get("description", "")
                        }
                    else:
                        processed_props[key] = {
                            "type": val.get("type", "string"),
                            "default": val.get("default", ""),
                            "description": val.get("description", "")
                        }

                function_declarations.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": {
                        "type": parameters.get("type", "object"),
                        "properties": processed_props,
                        "required": parameters.get("required", [])
                    }
                })

            payload["tools"] = [{"functionDeclarations": function_declarations}]

        print("\n=== [4] FINAL PAYLOAD TO GEMINI ===")
        print(json.dumps(payload, indent=2))

        # Send request to Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={params.api_key}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            
            print("\n=== [5] GEMINI RESPONSE ===")
            print(json.dumps(response_data, indent=2))

        except requests.exceptions.RequestException as req_err:
            err_data = {
                "message": "API request failed",
                "error": str(req_err),
                "status_code": getattr(req_err.response, 'status_code', None),
                "response": getattr(req_err.response, 'text', None)
            }
            return LlmResponseStruct(Data=None, Error=err_data, Status=False)

        # Process response
        message_content = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        tool_call = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("functionCall", None)
        is_tool_call = tool_call is not None

        usage = response_data.get("usageMetadata", {})

        final_format = SuccessResponseDataFormat(
            total_llm_calls=1,
            total_tokens=usage.get("totalTokenCount", 0),
            total_input_tokens=usage.get("promptTokenCount", 0),
            total_output_tokens=usage.get("candidatesTokenCount", 0),
            final_llm_response=response_data,
            llm_responses_arr=[response_data],
            messages=[message_content],
            output_type="tool_call" if is_tool_call else "text"
        )

        return LlmResponseStruct(Data=asdict(final_format), Error=None, Status=True)

    except Exception as err:
        return LlmResponseStruct(
            Data=None,
            Error={
                "message": "Unexpected processing error",
                "error": str(err),
                "type": type(err).__name__
            },
            Status=False
        )