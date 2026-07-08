import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Message = {
  id: number
  author: 'assistant' | 'user'
  text: string
  timestamp: string
}

const suggestions = [
  'Summarize results.json',
  'Find failed tasks',
  'Draft a follow-up prompt',
]

function getTimestamp() {
  return new Intl.DateTimeFormat('en', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(new Date())
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [isThinking, setIsThinking] = useState(false)

  const canSend = draft.trim().length > 0 && !isThinking
  const hasMessages = messages.length > 0

  function queueAssistantReply(userText: string) {
    setIsThinking(true)

    window.setTimeout(() => {
      setMessages((current) => [
        ...current,
        {
          id: Date.now(),
          author: 'assistant',
          text: `Got it. For "${userText}", I would pull the relevant context, compare it against the task requirements, and return the smallest useful summary with next actions.`,
          timestamp: getTimestamp(),
        },
      ])
      setIsThinking(false)
    }, 650)
  }

  function sendMessage(text = draft) {
    const trimmedText = text.trim()
    if (!trimmedText || isThinking) return

    setMessages((current) => [
      ...current,
      {
        id: Date.now(),
        author: 'user',
        text: trimmedText,
        timestamp: getTimestamp(),
      },
    ])
    setDraft('')
    queueAssistantReply(trimmedText)
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    sendMessage()
  }

  return (
    <main className="chat-shell">
      <section className="chat-panel" aria-label="Chat conversation">
        <div className="message-list" aria-live="polite">
          {!hasMessages && !isThinking && (
            <div className="welcome-state">
              <div>
                <p className="eyebrow">Workspace assistant</p>
                <h1>How can I help?</h1>
              </div>

              <div className="suggestions" aria-label="Suggested prompts">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => sendMessage(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <article
              key={message.id}
              className={`message message-${message.author}`}
            >
              <div className="message-avatar" aria-hidden="true">
                {message.author === 'assistant' ? 'AI' : 'You'}
              </div>
              <div className="message-content">
                <div className="message-meta">
                  <strong>
                    {message.author === 'assistant' ? 'Assistant' : 'You'}
                  </strong>
                  <time>{message.timestamp}</time>
                </div>
                <p>{message.text}</p>
              </div>
            </article>
          ))}

          {isThinking && (
            <article className="message message-assistant">
              <div className="message-avatar" aria-hidden="true">
                AI
              </div>
              <div className="message-content typing">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label htmlFor="message-input">Message</label>
          <div className="composer-row">
            <textarea
              id="message-input"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask about tasks, results, or next steps..."
              rows={2}
            />
            <button type="submit" disabled={!canSend}>
              Send
            </button>
          </div>
        </form>
      </section>
    </main>
  )
}

export default App
