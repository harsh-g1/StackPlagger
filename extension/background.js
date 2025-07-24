chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "analyze_code") {
    fetch("http://localhost:5000/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: msg.code, question: { tags: msg.tags || [] } }),
    })
      .then((res) => res.json())
      .then((data) => {
        sendResponse(data);
        setTimeout(() => {}, 1000); // keep alive
      })
      .catch((err) => {
        console.error("Backend error:", err);
        sendResponse({ error: "Failed to connect to backend" });
      });

    return true;
  }
});
