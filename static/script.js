function sendMessage() {
    let userMessage = document.getElementById("userInput").value;
    let chatBox = document.getElementById("messages");

    if (userMessage.trim() === "") return;
    
    chatBox.innerHTML += `<p><b>You:</b> ${userMessage}</p>`;

    fetch("/chat", {
        method: "POST",
        body: JSON.stringify({ query: userMessage }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<p><b>Chatbot:</b> ${data.response}</p>`;
        document.getElementById("userInput").value = "";
    });
}
