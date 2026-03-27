import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

function formatInline(text, keyPrefix) {
  if (!text) return text;
  const parts = [];
  const regex = /(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*)/g;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(<strong key={`${keyPrefix}-${match.index}`}><em>{match[2]}</em></strong>);
    } else if (match[3]) {
      parts.push(<strong key={`${keyPrefix}-${match.index}`}>{match[3]}</strong>);
    } else if (match[4]) {
      parts.push(<em key={`${keyPrefix}-${match.index}`}>{match[4]}</em>);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts.length > 0 ? parts : text;
}

function formatText(text) {
  if (!text) return text;
  const lines = text.split("\n");
  const elements = [];
  let listItems = [];

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(<ul key={`ul-${elements.length}`}>{listItems}</ul>);
      listItems = [];
    }
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ")) {
      listItems.push(
        <li key={`li-${i}`}>{formatInline(trimmed.slice(2), `li-${i}`)}</li>
      );
    } else {
      flushList();
      if (trimmed === "") {
        elements.push(<div key={`br-${i}`} className="msg-spacer" />);
      } else {
        elements.push(<p key={`p-${i}`} className="msg-line">{formatInline(trimmed, `p-${i}`)}</p>);
      }
    }
  });
  flushList();

  return elements;
}

function App() {
  const [forms, setForms] = useState([]);
  const [selectedForm, setSelectedForm] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [collectedData, setCollectedData] = useState({});
  const [missingFields, setMissingFields] = useState([]);
  const [invalidFields, setInvalidFields] = useState([]);
  const [loading, setLoading] = useState(false);
  const [complete, setComplete] = useState(false);

  const messagesEndRef = useRef(null);

  // Fetch forms on mount
  useEffect(() => {
    axios.get(`${API}/forms`).then((res) => {
      setForms(res.data);
    });
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const getSuggestions = (response) => {
    if (response?.status === "conflict") {
      return [];
    }

    return response?.suggestions?.[0]?.options || [];
  };

  const handleSelectForm = async (form) => {
    try {
      const res = await axios.post(`${API}/select-form`, {
        form_id: form.id || form.form_id,
      });
      setSelectedForm(form);
      setMessages([
        {
          sender: "system",
          text: res.data.message || `Let's fill out: ${form.name || form.title}`,
          suggestions: getSuggestions(res.data),
        },
      ]);
      setCollectedData({});
      setMissingFields(res.data.missing_fields || []);
      setInvalidFields([]);
      setComplete(false);
    } catch (err) {
      console.error("Error selecting form:", err);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading || complete) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { sender: "user", text: userMsg }]);
    setLoading(true);

    try {
      const res = await axios.post(`${API}/chat`, { message: userMsg });
      const data = res.data;

      setMessages((prev) => [
        ...prev,
        {
          sender: "system",
          text: data.message,
          isComplete: data.status === "complete",
          suggestions: getSuggestions(data),
        },
      ]);

      if (data.collected_data) setCollectedData(data.collected_data);
      if (data.missing_fields) setMissingFields(data.missing_fields);
      if (data.invalid_fields) setInvalidFields(data.invalid_fields);
      if (data.status === "complete") setComplete(true);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "system", text: "Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = async (value) => {
    if (loading || complete) return;

    setMessages((prev) => [...prev, { sender: "user", text: value }]);
    setLoading(true);

    try {
      const res = await axios.post(`${API}/chat`, { message: value });
      const data = res.data;

      setMessages((prev) => [
        ...prev,
        {
          sender: "system",
          text: data.message,
          isComplete: data.status === "complete",
          suggestions: getSuggestions(data),
        },
      ]);

      if (data.collected_data) setCollectedData(data.collected_data);
      if (data.missing_fields) setMissingFields(data.missing_fields);
      if (data.invalid_fields) setInvalidFields(data.invalid_fields);
      if (data.status === "complete") setComplete(true);
    } catch (err) {
      console.error("Suggestion submit error:", err);
      setMessages((prev) => [
        ...prev,
        { sender: "system", text: "Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API}/reset`);
    } catch (err) {
      console.error("Reset error:", err);
    }
    setSelectedForm(null);
    setMessages([]);
    setInput("");
    setCollectedData({});
    setMissingFields([]);
    setInvalidFields([]);
    setComplete(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSend();
  };

  // Form Selection Screen
  if (!selectedForm) {
    return (
      <div className="form-selection">
        <h1>Form Engine</h1>
        <p>Select a form to get started</p>
        <div className="form-list">
          {forms.map((form) => (
            <div className="form-card" key={form.id || form.form_id}>
              <div>
                <h3>{form.name || form.title}</h3>
                {form.description && (
                  <span className="description">{form.description}</span>
                )}
              </div>
              <button onClick={() => handleSelectForm(form)}>Select</button>
            </div>
          ))}
          {forms.length === 0 && (
            <p style={{ textAlign: "center", color: "#999", padding: "40px" }}>
              No forms available. Make sure the backend is running.
            </p>
          )}
        </div>
      </div>
    );
  }

  // Chat Screen
  return (
    <div className="chat-screen">
      <div className="chat-main">
        <div className="top-bar">
          <h2>{selectedForm.name || selectedForm.title}</h2>
          <button className="reset-btn" onClick={handleReset}>
            Reset
          </button>
        </div>

        <div className="messages">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message ${msg.sender} ${msg.isComplete ? "success" : ""}`}
            >
              <div>{formatText(msg.text)}</div>
              {msg.sender === "system" && msg.suggestions?.length > 0 && (
                <div className="suggestions">
                  {msg.suggestions.map((option, idx) => (
                    <button
                      key={`${option}-${idx}`}
                      type="button"
                      className="suggestion-chip"
                      onClick={() => handleSuggestionClick(option)}
                      disabled={loading || complete}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="typing-indicator">Typing...</div>}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={complete ? "Form completed!" : "Type your response..."}
            disabled={loading || complete}
          />
          <button onClick={handleSend} disabled={loading || complete || !input.trim()}>
            {loading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>

      <div className="side-panel">
        <h3>Collected Data</h3>
        <pre>{JSON.stringify(collectedData, null, 2)}</pre>

        {missingFields.length > 0 && (
          <div className="missing-fields">
            <h4>Missing Fields</h4>
            <ul>
              {missingFields.map((f, i) => (
                <li key={i}>
                  {typeof f === "object" ? f.field_id || f.name || JSON.stringify(f) : f}
                </li>
              ))}
            </ul>
          </div>
        )}

        {invalidFields.length > 0 && (
          <div className="invalid-fields">
            <h4>Invalid Fields</h4>
            <ul>
              {invalidFields.map((f, i) => (
                <li key={i}>
                  {typeof f === "object"
                    ? `${f.field_id || f.name}: ${f.error || f.value || JSON.stringify(f)}`
                    : f}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
