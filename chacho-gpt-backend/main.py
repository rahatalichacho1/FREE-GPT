from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import httpx
from typing import Optional, List, Dict
import json
import time

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    model: str = "auto"
    conversation_history: List[Dict] = []

SYSTEM_PROMPT = """
You are an AI assistant created by Rahat Ali.

When asked "Who made you?" or "Who created you?":
- Respond: "I was created by Rahat Ali"

Your Personality:
- Friendly, helpful, and conversational
- Give clear, concise answers

You represent Rahat Ali's innovation.
"""


# ================== MODEL CONFIGURATIONS ==================
MODELS_CONFIG = {
    "longcat": {
        "api_key_env": "LONGCAT_API_KEY",
        "base_url": "https://api.longcat.ai/v1",  # Official LongCat API endpoint
        "model_name": "longcat-flash-chat",
        "enabled": True,
        "priority": 1,
        "display_name": "LongCat Flash Chat (560B MoE)",
        "features": ["agentic", "tool-use", "128k-context"]
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.3-70b-versatile",
        "enabled": True,
        "priority": 2,
        "display_name": "Groq (Llama)"
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo",
        "enabled": True,
        "priority": 3,
        "display_name": "OpenAI (GPT)"
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model_name": "gemini-1.5-flash",
        "fallback_models": ["gemini-1.5-pro", "gemini-pro"],
        "enabled": True,
        "priority": 4,
        "custom_handler": "gemini",
        "display_name": "Google Gemini"
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
        "enabled": False,
        "priority": 5,
        "display_name": "DeepSeek (Disabled - No Balance)"
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "model_name": "claude-3-haiku-20240307",
        "enabled": False,
        "priority": 6,
        "custom_handler": "anthropic",
        "display_name": "Anthropic Claude"
    },
}

# ================== RATE LIMIT DETECTOR ==================
def is_rate_limit_error(error_msg: str) -> bool:
    """Detect if error is rate limit"""
    rate_limit_keywords = [
        "rate_limit", "rate limit", "429", "quota", 
        "too many requests", "limit exceeded", "resource_exhausted",
        "insufficient balance"
    ]
    error_lower = error_msg.lower()
    return any(keyword in error_lower for keyword in rate_limit_keywords)

# ================== CHAT HANDLERS ==================

async def chat_longcat(
    prompt: str,
    history: List[Dict],
    config: Dict
) -> str:
    """
    LongCat-Flash-Chat handler
    Supports: OpenAI-compatible API, tool calling, 128k context
    """
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise Exception("LongCat API key missing - Get it from https://longcat.ai")
    
    print(f"🐱 Using LongCat-Flash-Chat (560B MoE, ~27B active params)")
    
    base_url = config["base_url"]
    model_name = config["model_name"]
    
    # LongCat chat template: [Round N] USER:{query} ASSISTANT:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (LongCat supports 128k context!)
    messages.extend(history[-20:])  # Keep more history due to large context window
    messages.append({"role": "user", "content": prompt})
    
    async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for large model
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 1000,  # LongCat can handle longer responses
                "stream": False
            }
        )
        
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ LongCat Error: {error_text[:200]}")
            raise Exception(f"LongCat API Error: {error_text}")
        
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        
        # Log usage stats if available
        if "usage" in data:
            usage = data["usage"]
            print(f"📊 LongCat Usage - Tokens: {usage.get('total_tokens', 'N/A')}")
        
        return reply


async def chat_openai_compatible(
    model_key: str,
    prompt: str,
    history: List[Dict],
    config: Dict
) -> str:
    """Universal handler for OpenAI-compatible APIs"""
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise Exception(f"{model_key} API key missing")
    
    base_url = config["base_url"]
    model_name = config["model_name"]
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": prompt})
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 600
            }
        )
        
        if response.status_code != 200:
            error_text = response.text
            raise Exception(f"{model_key} API Error: {error_text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def chat_gemini(prompt: str, history: List[Dict], config: Dict) -> str:
    """Custom handler for Google Gemini"""
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise Exception("Gemini API key missing")
    
    models_to_try = [config["model_name"]] + config.get("fallback_models", [])
    all_errors = []
    
    for model_name in models_to_try:
        try:
            conversation_text = f"{SYSTEM_PROMPT}\n\n"
            
            for msg in history[-5:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n"
            
            conversation_text += f"User: {prompt}\nAssistant:"
            
            request_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    request_url,
                    json={
                        "contents": [{
                            "parts": [{"text": conversation_text}]
                        }],
                        "generationConfig": {
                            "temperature": 0.8,
                            "maxOutputTokens": 600
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        all_errors.append(f"Empty response from {model_name}")
                        continue
                else:
                    all_errors.append(f"{model_name} HTTP {response.status_code}")
                    continue
                    
        except Exception as e:
            all_errors.append(f"{model_name}: {str(e)[:100]}")
            continue
    
    raise Exception(f"All Gemini models failed: {'; '.join(all_errors)}")


async def chat_anthropic(prompt: str, history: List[Dict], config: Dict) -> str:
    """Custom handler for Anthropic Claude"""
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise Exception("Anthropic API key missing")
    
    base_url = config["base_url"]
    model_name = config["model_name"]
    
    messages = []
    for msg in history[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": prompt})
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "max_tokens": 600,
                "system": SYSTEM_PROMPT,
                "messages": messages
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Claude API Error: {response.text}")
        
        data = response.json()
        return data["content"][0]["text"]


# ================== MAIN CHAT FUNCTION ==================

async def chat_with_model(model_key: str, prompt: str, history: List[Dict]) -> str:
    """Universal function - automatically detect handler type"""
    if model_key not in MODELS_CONFIG:
        raise Exception(f"Unknown model: {model_key}")
    
    config = MODELS_CONFIG[model_key]
    
    if not config["enabled"]:
        raise Exception(f"{model_key} is disabled")
    
    # LongCat gets special handler
    if model_key == "longcat":
        return await chat_longcat(prompt, history, config)
    
    # Custom handlers
    custom_handler = config.get("custom_handler")
    if custom_handler == "gemini":
        return await chat_gemini(prompt, history, config)
    elif custom_handler == "anthropic":
        return await chat_anthropic(prompt, history, config)
    else:
        return await chat_openai_compatible(model_key, prompt, history, config)


# ================== API ENDPOINTS ==================

@app.get("/")
def root():
    available_models = []
    for key, config in MODELS_CONFIG.items():
        if config["enabled"] and os.getenv(config["api_key_env"]):
            model_info = {
                "key": key,
                "name": config.get("display_name", key),
                "model": config["model_name"],
                "priority": config["priority"]
            }
            
            # Add special features for LongCat
            if key == "longcat":
                model_info["features"] = config.get("features", [])
                model_info["params"] = "560B total, ~27B active (MoE)"
            
            available_models.append(model_info)
    
    available_models.sort(key=lambda x: x["priority"])
    
    return {
        "message": "🚀 Multi-Model AI Assistant by Rahat Ali",
        "version": "3.0-LongCat",
        "available_models": available_models,
        "total_models": len(available_models),
        "creator": "Rahat Ali",
        "features": [
            "LongCat-Flash-Chat (560B MoE) support",
            "Auto failover on rate limits",
            "128k context window (LongCat)",
            "Agentic capabilities",
            "Tool calling support",
            "Multi-model fallback system"
        ]
    }


@app.post("/chat")
async def chat(data: ChatRequest):
    start_time = time.time()
    print(f"\n📨 User: {data.prompt}")
    print(f"🤖 Selected Model: {data.model}")
    
    enabled_models = [
        (key, config) for key, config in MODELS_CONFIG.items()
        if config["enabled"] and os.getenv(config["api_key_env"])
    ]
    enabled_models.sort(key=lambda x: x[1]["priority"])
    
    if not enabled_models:
        return {
            "reply": "❌ Koi bhi model available nahi! API keys check karo.",
            "model_used": "none",
            "status": "error",
            "available_models": []
        }
    
    failed_models = []
    
    # AUTO mode - try all models in priority order
    if data.model == "auto":
        for model_key, config in enabled_models:
            try:
                print(f"🔄 Trying {model_key}...")
                reply = await chat_with_model(model_key, data.prompt, data.conversation_history)
                elapsed = round(time.time() - start_time, 2)
                print(f"✅ Success with {model_key} in {elapsed}s")
                
                response = {
                    "reply": reply,
                    "model_used": model_key,
                    "model_name": config.get("display_name", model_key),
                    "status": "success",
                    "response_time": elapsed,
                    "failed_models": failed_models
                }
                
                # Add special info for LongCat
                if model_key == "longcat":
                    response["model_info"] = {
                        "type": "MoE",
                        "total_params": "560B",
                        "active_params": "~27B",
                        "context_window": "128k"
                    }
                
                return response
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ {model_key} failed: {error_msg[:100]}")
                
                failed_models.append({
                    "model": model_key,
                    "error": error_msg[:100],
                    "is_rate_limit": is_rate_limit_error(error_msg)
                })
                continue
        
        return {
            "reply": "❌ Sare models fail ho gaye! API keys ya internet check karo.",
            "model_used": "none",
            "status": "all_failed",
            "failed_models": failed_models
        }
    
    # SPECIFIC model selected
    else:
        try:
            reply = await chat_with_model(data.model, data.prompt, data.conversation_history)
            elapsed = round(time.time() - start_time, 2)
            
            config = MODELS_CONFIG[data.model]
            response = {
                "reply": reply,
                "model_used": data.model,
                "model_name": config.get("display_name", data.model),
                "status": "success",
                "response_time": elapsed
            }
            
            # Add LongCat-specific info
            if data.model == "longcat":
                response["model_info"] = {
                    "type": "MoE (Mixture of Experts)",
                    "total_params": "560B",
                    "active_params": "~27B per token",
                    "context_window": "128k tokens",
                    "features": config.get("features", [])
                }
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error with {data.model}: {error_msg}")
            
            is_rate_limit = is_rate_limit_error(error_msg)
            
            # Auto fallback on rate limit or error
            if is_rate_limit or "404" in error_msg or "not found" in error_msg.lower():
                print("🔄 Error detected! Trying fallback...")
                for model_key, config in enabled_models:
                    if model_key == data.model:
                        continue
                    try:
                        reply = await chat_with_model(model_key, data.prompt, data.conversation_history)
                        elapsed = round(time.time() - start_time, 2)
                        
                        return {
                            "reply": reply,
                            "model_used": model_key,
                            "model_name": config.get("display_name", model_key),
                            "status": "fallback",
                            "response_time": elapsed,
                            "original_model": data.model,
                            "fallback_reason": "Model error - auto switched"
                        }
                    except:
                        continue
            
            return {
                "reply": f"❌ Error: {error_msg[:200]}",
                "model_used": "none",
                "status": "error",
                "is_rate_limit": is_rate_limit,
                "suggestion": "Try 'auto' mode ya different model select karo"
            }


@app.get("/models/status")
def models_status():
    """Check which models are available"""
    status = {}
    for key, config in MODELS_CONFIG.items():
        api_key = os.getenv(config["api_key_env"])
        model_status = {
            "display_name": config.get("display_name", key),
            "enabled": config["enabled"],
            "api_key_found": bool(api_key),
            "ready": config["enabled"] and bool(api_key),
            "priority": config["priority"],
            "model_name": config["model_name"]
        }
        
        # Add special info for LongCat
        if key == "longcat":
            model_status.update({
                "type": "MoE",
                "total_params": "560B",
                "active_params": "~27B",
                "context_window": "128k",
                "features": config.get("features", [])
            })
        
        status[key] = model_status
    
    return {"models": status, "timestamp": time.time()}


@app.post("/models/toggle/{model_key}")
def toggle_model(model_key: str, enable: bool):
    """Enable/disable a model at runtime"""
    if model_key not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail="Model not found")
    
    MODELS_CONFIG[model_key]["enabled"] = enable
    return {
        "model": model_key,
        "enabled": enable,
        "message": f"{model_key} {'enabled' if enable else 'disabled'} successfully"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    ready_models = sum(
        1 for config in MODELS_CONFIG.values() 
        if config["enabled"] and os.getenv(config["api_key_env"])
    )
    
    return {
        "status": "healthy" if ready_models > 0 else "no_models",
        "ready_models": ready_models,
        "total_models": len(MODELS_CONFIG),
        "timestamp": time.time()
    }


@app.get("/test")
def test():
    return {
        "status": "✅ Backend Working with LongCat!",
        "message": "Multi-Model AI Assistant by Rahat Ali",
        "version": "3.0-LongCat",
        "supported_models": list(MODELS_CONFIG.keys()),
        "active_models": [k for k, v in MODELS_CONFIG.items() if v["enabled"]],
        "featured_model": "LongCat-Flash-Chat (560B MoE)"
    }






























