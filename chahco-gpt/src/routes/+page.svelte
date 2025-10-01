<script>
  import "./page.css";
  import { welcome_message } from "../message.js";
  import { onMount } from "svelte";

  let welcome_text = "";
  let show_input = false;
  let user_input = "";
  let messages = [];
  let loading = false;
  let chatContainer;
  
  // Multi-model support
  let selectedModel = "auto";
  let availableModels = [];
  let currentModelUsed = "";
  let showModelSelector = false;
  
  const BACKEND_URL = "http://localhost:8000";

  onMount(async () => {
    // Welcome message animation
    setTimeout(() => {
      welcome_text = welcome_message[Math.floor(Math.random() * welcome_message.length)];
    }, 50);
    
    setTimeout(() => {
      show_input = true;
    }, 60);

    // Fetch available models
    await fetchAvailableModels();
  });

  async function fetchAvailableModels() {
    try {
      const res = await fetch(`${BACKEND_URL}/`);
      if (res.ok) {
        const data = await res.json();
        availableModels = data.available_models || [];
        console.log("✅ Available models:", availableModels);
      }
    } catch (error) {
      console.error("❌ Error fetching models:", error);
    }
  }

  function scrollToBottom() {
    setTimeout(() => {
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 100);
  }

  async function submit_input() {
    const trimmedInput = user_input.trim();
    if (trimmedInput === "" || loading) return;

    // Add user message
    messages = [
      ...messages,
      {
        role: "user",
        content: trimmedInput,
      },
    ];

    const currentPrompt = user_input;
    user_input = "";
    loading = true;
    scrollToBottom();

    // Prepare conversation history for backend
    const conversationHistory = messages
      .slice(0, -1) // Exclude current message
      .map(msg => ({
        role: msg.role === "user" ? "user" : "assistant",
        content: msg.content
      }));

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: currentPrompt,
          model: selectedModel,
          conversation_history: conversationHistory
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      
      // Update model used
      currentModelUsed = data.model_used || selectedModel;

      // Add assistant message
      let assistantContent = data.reply || "Sorry, something went wrong";
      
      // Add model info if fallback happened
      if (data.status === "fallback") {
        assistantContent += `\n\n_[Switched from ${data.original_model} to ${data.model_used} - ${data.fallback_reason}]_`;
      }

      messages = [
        ...messages,
        {
          role: "assistant",
          content: assistantContent,
          modelUsed: data.model_used,
          responseTime: data.response_time
        },
      ];

    } catch (error) {
      console.error("❌ Error:", error);
      messages = [
        ...messages,
        {
          role: "assistant",
          content: `❌ Backend connection error!\n\nMake sure backend is running:\n\`uvicorn main:app --reload\`\n\nURL: ${BACKEND_URL}`,
        },
      ];
    } finally {
      loading = false;
      scrollToBottom();
    }
  }

  function handleKeyPress(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit_input();
    }
  }

  function clearChat() {
    messages = [];
    currentModelUsed = "";
    welcome_text = welcome_message[Math.floor(Math.random() * welcome_message.length)];
  }

  function toggleModelSelector() {
    showModelSelector = !showModelSelector;
  }

  function selectModel(modelKey) {
    selectedModel = modelKey;
    showModelSelector = false;
  }
</script>

<main>
  <div class="container">
    
    <!-- Model Selector Dropdown -->
    {#if show_input && availableModels.length > 0}
      <div class="model-selector-container">
        <button class="model-selector-btn" on:click={toggleModelSelector}>
          <span class="model-icon">🤖</span>
          <span class="model-name">
            {#if selectedModel === "auto"}
              Auto (Smart Selection)
            {:else}
              {availableModels.find(m => m.key === selectedModel)?.name || selectedModel}
            {/if}
          </span>
          <span class="dropdown-arrow">{showModelSelector ? "▲" : "▼"}</span>
        </button>
        
        {#if showModelSelector}
          <div class="model-dropdown">
            <button 
              class="model-option" 
              class:active={selectedModel === "auto"}
              on:click={() => selectModel("auto")}
            >
              <span class="model-option-icon">⚡</span>
              <div class="model-option-content">
                <div class="model-option-name">Auto Mode</div>
                <div class="model-option-desc">Smart failover enabled</div>
              </div>
            </button>
            
            {#each availableModels as model}
              <button 
                class="model-option"
                class:active={selectedModel === model.key}
                on:click={() => selectModel(model.key)}
              >
                <span class="model-option-icon">🔹</span>
                <div class="model-option-content">
                  <div class="model-option-name">{model.name}</div>
                  <div class="model-option-desc">Priority: {model.priority}</div>
                </div>
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Welcome Text -->
    {#if welcome_text && messages.length === 0}
      <h1 class="welcome_text">{welcome_text}</h1>
    {/if}

    <!-- Chat Container -->
    {#if messages.length > 0}
      <div class="chat-header">
        <button class="clear-btn" on:click={clearChat}>
          🗑️ Clear Chat
        </button>
      </div>

      <div class="chat-container" bind:this={chatContainer}>
        {#each messages as message, i}
          <div class="message {message.role}">
            <div class="message-content">
              {@html message.content.replace(/\n/g, '<br>')}
              
              {#if message.modelUsed && message.responseTime}
                <div class="message-meta">
                  <span class="meta-model">🤖 {message.modelUsed}</span>
                  <span class="meta-time">⏱️ {message.responseTime}s</span>
                </div>
              {/if}
            </div>
          </div>
        {/each}

        <!-- Typing Indicator -->
        {#if loading}
          <div class="message assistant">
            <div class="message-content">
              <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
              {#if currentModelUsed}
                <div class="typing-model">Using {currentModelUsed}...</div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Input Container -->
    {#if show_input}
      <div class="input-container">
        <input
          type="text"
          placeholder="Ask me anything :)"
          bind:value={user_input}
          on:keypress={handleKeyPress}
          disabled={loading}
          autofocus
        />
        <button
          on:click={submit_input}
          disabled={loading || user_input.trim() === ""}
          class:loading
        >
          {loading ? "●●●" : "Ask"}
        </button>
      </div>
    {/if}
  </div>
</main>