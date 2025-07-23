document.addEventListener("DOMContentLoaded", function () {
  const infoSection = document.getElementById("infoSection");
  infoSection.innerHTML = `
    <h3>AI Code Analyzer</h3>
    <p>Automatically analyzing code in Stack Overflow answers...</p>
    <p class="small">Confidence scores appear below each answer.</p>
  `;
});