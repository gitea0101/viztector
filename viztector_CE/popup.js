function removeAllEventListeners() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: () => {
        resetHighlight();
        savedCSVData = ""
        removeAllLayers();
      }
    });
  });
}

document.getElementById('start').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: () => {
        startImageSelection();
      }
    });
  });
  document.getElementById('capture').disabled = false;
  document.getElementById('capture').style.backgroundColor = 'rgb(255, 121, 0)';
});

document.getElementById('reset').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: () => {
        resetHighlight();
        savedCSVData = ""
        removeAllLayers();
      }
    });
  });
  document.getElementById('capture').disabled = true;
  document.getElementById('capture').style.backgroundColor = '#d3d3d3';
});

document.getElementById('capture').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: () => {
        startCaptureProcess();
      }
    });
  });
});

document.getElementById('close').addEventListener('click', () => {
  chrome.runtime.sendMessage({ action: 'closeExtension' }); // 확장 프로그램 종료
  window.close()
});

document.getElementById('download').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: () => {
        downloadCSV(savedCSVData);
      }
    });
  });
});

