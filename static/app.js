const emailInput = document.getElementById("emailInput");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const confidenceText = document.getElementById("confidenceText");
const themeToggle = document.getElementById("themeToggle");

// 🌓 Dark / Light Mode Toggle Logic
function updateThemeButton(isDark) {
  themeToggle.textContent = isDark ? "☀️ Light Mode" : "🌙 Dark Mode";
}

// Load saved theme preference
const savedTheme = localStorage.getItem("theme");
if (savedTheme === "dark") {
  document.body.classList.add("dark-mode");
  updateThemeButton(true);
} else {
  updateThemeButton(false);
}

themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark-mode");
  const isDark = document.body.classList.contains("dark-mode");
  localStorage.setItem("theme", isDark ? "dark" : "light");
  updateThemeButton(isDark);
});

// Check email when button is clicked
predictBtn.addEventListener("click", async () => {
  const text = emailInput.value.trim();

  if (!text) {
    alert("Please enter some email text first!");
    return;
  }

  predictBtn.textContent = "Checking...";
  predictBtn.disabled = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    // Show result box
    resultBox.classList.remove("hidden");

    if (data.is_spam) {
      resultText.textContent = "🚨 SPAM EMAIL";
      resultText.className = "result-text spam";
    } else {
      resultText.textContent = "✅ LEGITIMATE (HAM) EMAIL";
      resultText.className = "result-text ham";
    }

    confidenceText.textContent = `Confidence: ${data.confidence}% (Ham: ${data.probabilities.ham}%, Spam: ${data.probabilities.spam}%)`;
  } catch (error) {
    alert("Error making prediction: " + error.message);
  } finally {
    predictBtn.textContent = "Check Email";
    predictBtn.disabled = false;
  }
});

// Clear input
clearBtn.addEventListener("click", () => {
  emailInput.value = "";
  resultBox.classList.add("hidden");
});

// Load sample emails
function loadSample(type) {
  if (type === 1) {
    emailInput.value = "Hi team,\nCan we reschedule our project sync meeting to 3 PM tomorrow?\nThanks,\nAlex";
  } else {
    emailInput.value = "CONGRATULATIONS! You have won $1,000,000 in our cash lottery prize. Claim your free money now by replying immediately!";
  }
}
