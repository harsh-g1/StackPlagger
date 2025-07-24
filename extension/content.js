function getAllAnswersWithCode() {
  const answers = document.querySelectorAll('.answer:not(.question)');
  if (!answers || answers.length === 0) return { error: "NO_ANSWERS_FOUND" };

  const results = [];
  answers.forEach((answer) => {
    const codeBlocks = answer.querySelectorAll('pre code, .s-code-block');
    if (!codeBlocks || codeBlocks.length === 0) return;

    let combinedCode = '';
    codeBlocks.forEach((block, index) => {
      if (index > 0) combinedCode += '\n\n// ----- Code Block Separator -----\n\n';
      combinedCode += block.textContent.trim();
    });

    if (combinedCode.length < 50) return;

    results.push({
      answerElement: answer,
      code: combinedCode,
      isVerified: answer.classList.contains('accepted-answer') || 
                 answer.classList.contains('js-accepted-answer'),
      author: answer.querySelector('.user-info .user-details a')?.textContent.trim() || 'Unknown',
      votes: answer.querySelector('.js-vote-count')?.textContent.trim() || '0',
      codeBlockCount: codeBlocks.length,
      positionElement: answer.querySelector('.post-layout') || answer.querySelector('.answercell') || answer
    });
  });

  if (results.length === 0) return { error: "NO_CODE_IN_ANSWERS" };
  return { answers: results };
}

function displayConfidenceScore(answerElement, positionElement, score, isAI, isVerified) {
  // Remove existing score if present
  const existingScore = answerElement.querySelector('.ai-confidence-display');
  if (existingScore) existingScore.remove();

  // Create score display element
  const scoreElement = document.createElement('div');
  scoreElement.className = `ai-confidence-display ${isAI ? 'ai-detected' : 'human-written'} ${
    isVerified ? 'verified-answer' : ''
  }`;
  scoreElement.innerHTML = `
    <span class="ai-confidence-label">AI Confidence:</span>
    <span class="ai-confidence-value">${score}%</span>
    ${isVerified ? '<span class="verified-badge"></span>' : ''}
  `;

  // Positioning the element
  if (positionElement) {
    const header = positionElement.querySelector('.post-text') || positionElement;
    if (header) {
      header.insertAdjacentElement('beforebegin', scoreElement);
    } else {
      positionElement.insertAdjacentElement('afterbegin', scoreElement);
    }
  }
}

// Adding styles for the confidence display
const style = document.createElement('style');
style.textContent = `
  .ai-confidence-display {
    padding: 8px 12px;
    margin: 10px 0;
    border-radius: 4px;
    font-size: 13px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    width: fit-content;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .ai-confidence-label {
    font-weight: 500;
  }
  .ai-confidence-value {
    font-weight: bold;
  }
  .ai-detected {
    background-color: #fee2e2;
    color: #b91c1c;
    border-left: 3px solid #ef4444;
  }
  .human-written {
    background-color: #dcfce7;
    color: #166534;
    border-left: 3px solid #10b981;
  }
  .verified-badge {
    margin-left: 8px;
    padding: 2px 6px;
    background-color: #e6f3ff;
    color: #0064bd;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 500;
  }
  .verified-answer {
    border-left-width: 4px;
  }
`;
document.head.appendChild(style);
//-------------------------------------------------------
function getCurrentQuestionTags() {
  return Array.from(document.querySelectorAll('.question .post-tag'))
    .map(el => el.innerText.toLowerCase());
}
//----------------------------------------------------------
// Main function to analyze all answers
function analyzeAllAnswers() {
  const result = getAllAnswersWithCode();
  
  if (result.error) {
    console.log('AI Detection:', result.error); // logging
    return;
  }

  const questionTags = getCurrentQuestionTags();
  console.log("Current question tags:", questionTags); // Debug log

  result.answers.forEach((answer) => {
    chrome.runtime.sendMessage(
    {
      type: "analyze_code",
      code: answer.code,
      tags: questionTags,  
      source: "answer_code"
    },
    (response) => {
      if (!response.error) {
        const score = Math.round(response.ai_probability * 100);
        displayConfidenceScore(
          answer.answerElement,
          answer.positionElement,
          score,
          response.is_ai_generated,
          answer.isVerified
        );
      }
    }
    );
  });
}

// Run automatically when Stack Overflow page loads
if (window.location.hostname.includes('stackoverflow.com')) {
  // Wait for page to fully load
  if (document.readyState === 'complete') {
    setTimeout(analyzeAllAnswers, 2000);
  } else {
    window.addEventListener('load', () => {
      setTimeout(analyzeAllAnswers, 2000);
    });
  }

  // Also run when new answers might be loaded (e.g., infinite scroll)
  const observer = new MutationObserver(() => {
    if (document.querySelector('.answer:not(.question)')) {
      setTimeout(analyzeAllAnswers, 1000);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}
