let savedCSVData = [
  ["name", "age", "email"],
  ["John Doe", 29, "john@example.com"],
  ["Jane Smith", 34, "jane@example.com"]
];
savedCSVData = savedCSVData.map(e => e.join(",")).join("\n");
// Function to start image selection
function startImageSelection() {
  highlightImages();
}

// Function to highlight images with a red border
function highlightImages() {
  document.querySelectorAll('img').forEach((img) => {
    img.style.border = '2px solid red';
    img.addEventListener('click', handleClick);
  });
}
// Function to create the popup
function createPopup() {
  const popup = document.createElement('div');
  popup.id = 'popup';
  popup.innerHTML = `
    <button id="start" class="round-button">START</button>
    <button id="reset" class="round-button">RESET</button>
    <button id="capture" class="round-button" disabled>CAPTURE</button>
    <button id="close" class="close-button">X</button>
  `;
  document.body.appendChild(popup);
}

// Create the popup when the script is loaded
// createPopup();


// Function to reset highlights
function resetHighlight() {
  document.querySelectorAll('img').forEach((img) => {
    img.style.border = '';
    img.removeEventListener('click', handleClick);
  });
}

function handleClick(event) {
  event.preventDefault();
  event.stopPropagation();
  selectImage(event.target);
}

// Function to handle image selection
function selectImage(imgElement) {
  document.querySelectorAll('img').forEach((img) => {
    img.style.border = '';
    img.removeEventListener('click', handleClick);
  });
  imgElement.classList.add('selected-image');
  const rect = imgElement.getBoundingClientRect();
  showLoadingNextToElement(rect);
  sendImageUrlToServer(imgElement.src, rect);
  console.log("Image URL sent to server:", imgElement.src);
}

// Function to send image URL to the server via WebSocket (rect 인자 추가)
function sendImageUrlToServer(imageUrl, rect) {
  const socket = new WebSocket('ws://127.0.0.1:8080');

  socket.onopen = () => {
    console.log("WebSocket connection opened");
    socket.send(JSON.stringify({ imageUrl: imageUrl }));
  };

  socket.onmessage = (event) => {
    console.log("서버로부터 데이터 수신");

    const imageData = JSON.parse(event.data);

    if (imageData.error) {
      showLoadingError();
    } else {
      // 로딩 화면 제거
      const loadingLayer = document.getElementById('loading-layer');
      if (loadingLayer) {
        loadingLayer.remove();
      }
      if (imageData.correctionMark) {
        const markImageUrl = `data:image/png;base64,${imageData.correctionMark}`;
        showCorrectionMarkImage(markImageUrl);
      }
      if (imageData.correctionCopy) {
        const copyImageUrl = `data:image/png;base64,${imageData.correctionCopy}`;
        showImageNextToElement(copyImageUrl, rect);
      }
      if (typeof imageData.distortionDetected !== 'undefined') {
        showDistortionResult(imageData.distortionDetected);
      }
      if (imageData.csvFile) {
        savedCSVData = imageData.csvFile; // CSV 데이터를 변수에 저장
        downloadCSV(savedCSVData);
      }
    }
  };

  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
    showLoadingError(); //로딩화면에 오류 메시지 표시
  };

  socket.onclose = () => {
    console.log('WebSocket connection closed');
  };
}

// Function to send image data to the server via WebSocket
function sendImageDataToServer(base64ImageData, rect) {
  const socket = new WebSocket('ws://127.0.0.1:8080');

  socket.onopen = () => {
    console.log("WebSocket connection opened");
    socket.send(JSON.stringify({ imageData: base64ImageData }));
  };

  socket.onmessage = (event) => {
    console.log("서버로부터 데이터 수신");

    const imageData = JSON.parse(event.data);

    if (imageData.error) {
      showLoadingError();
    } else {
      // 로딩 화면 제거
      const loadingLayer = document.getElementById('loading-layer');
      if (loadingLayer) {
        loadingLayer.remove();
      }

      if (imageData.correctionMark) {
        const markImageUrl = `data:image/png;base64,${imageData.correctionMark}`;
        showCorrectionMarkImage(markImageUrl);
      }
      if (imageData.correctionCopy) {
        const copyImageUrl = `data:image/png;base64,${imageData.correctionCopy}`;
        showImageNextToElement(copyImageUrl, rect);
      }
      if (typeof imageData.distortionDetected !== 'undefined') {
        showDistortionResult(imageData.distortionDetected);
      }
      if (imageData.csvFile) {
        savedCSVData = imageData.csvFile; // CSV 데이터를 변수에 저장
        downloadCSV(savedCSVData);
      }
      if (imageData.error) {
        showLoadingError();
      }
    }
  };

  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
    showLoadingError(); //로딩화면에 오류 메시지 표시
  };

  socket.onclose = () => {
    console.log('WebSocket connection closed');
  };
}

function downloadCSV(csvContent) {
  const encodedUri = encodeURI(`data:text/csv;charset=utf-8,${csvContent}`);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "data.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function showDistortionResult(isDistorted) {
  const imageLayer = document.getElementById('image-layer');
  if (isDistorted !== true) {
    imageLayer.innerHTML = ''; // Clear existing loading animation
    const resultMessage = document.createElement('div');
    resultMessage.innerText = isDistorted ? '왜곡이 감지되었습니다.' : '왜곡이 감지되지 않았습니다.';
    resultMessage.style.color = 'white';
    resultMessage.style.fontSize = '16px';
    imageLayer.appendChild(resultMessage);
  }
}

function showCorrectionMarkImage(imageUrl) {
  const selectedImage = document.querySelector('.selected-image');
  const selectedVideo = document.querySelector('video');

  if (selectedImage) {
    selectedImage.src = imageUrl;
  } else if (selectedVideo) {
    const videoRect = selectedVideo.getBoundingClientRect();
    const overlay = document.createElement('img');
    overlay.src = imageUrl;
    overlay.style.position = 'absolute';
    overlay.style.top = `${videoRect.top + window.scrollY}px`;
    overlay.style.left = `${videoRect.left + window.scrollX}px`;
    overlay.style.width = `${videoRect.width}px`;
    overlay.style.height = `${videoRect.height}px`;
    overlay.style.zIndex = '10000';
    document.body.appendChild(overlay);
  } else {
    console.error('교정표시본을 대체할 이미지가 선택되지 않았습니다.');
  }
}

function showCorrectionCopyImage(imageUrl) {
  const selectedImage = document.querySelector('.selected-image');
  const selectedVideo = document.querySelector('video');
  if (selectedImage) {
    const rect = selectedImage.getBoundingClientRect();
    showImageNextToElement(imageUrl, rect);
  } else if (selectedVideo) {
    const rect = selectedVideo.getBoundingClientRect();
    showImageNextToElement(imageUrl, rect);
  } else {
    console.error('교정본을 표시할 이미지가 선택되지 않았습니다.');
  }
}

function showImageNextToElement(imageUrl, rect) {
  const layer = document.createElement('div');
  layer.id = 'image-layer';
  layer.style.position = 'absolute';
  layer.style.top = `${rect.top + window.scrollY}px`; // 클릭한 이미지의 절대 위쪽 위치
  layer.style.left = `${rect.right + window.scrollX + 10}px`; // 클릭한 이미지의 절대 오른쪽 위치
  layer.style.width = `${rect.width}px`; // 클릭한 이미지의 너비
  layer.style.height = `${rect.height}px`; // 클릭한 이미지의 높이
  layer.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
  layer.style.zIndex = '9999';
  layer.style.display = 'flex';
  layer.style.justifyContent = 'center';
  layer.style.alignItems = 'center';
  layer.style.overflow = 'hidden';

  const imageElement = document.createElement('img');
  imageElement.src = imageUrl;
  imageElement.style.maxWidth = '100%';
  imageElement.style.maxHeight = '100%';

  const closeButton = document.createElement('button');
  closeButton.innerText = 'X';
  closeButton.style.position = 'absolute';
  closeButton.style.top = '5px';
  closeButton.style.right = '5px';
  closeButton.style.padding = '5px';
  closeButton.style.backgroundColor = 'black';
  closeButton.style.color = 'white';
  closeButton.style.border = 'none';
  closeButton.style.borderRadius = '3px';
  closeButton.style.cursor = 'pointer';

  closeButton.addEventListener('click', () => {
    layer.remove();
  });

  layer.innerHTML = ''; // Clear loading content
  layer.appendChild(imageElement);
  layer.appendChild(closeButton);
  document.body.appendChild(layer);

  adjustCorrectionImagePosition(); // 위치 조정

  // Adjust position if out of viewport
  const layerRect = layer.getBoundingClientRect();

  // Ensure the layer is within the viewport horizontally
  if (layerRect.right > window.innerWidth) {
    layer.style.left = `${rect.left + window.scrollX - layerRect.width - 10}px`;
  }

  // Ensure the layer is within the viewport vertically (bottom boundary)
  if (layerRect.bottom > window.innerHeight) {
    layer.style.top = `${window.innerHeight - layerRect.height - 10 + window.scrollY}px`;
  }

  // Ensure the layer is within the viewport vertically (top boundary)
  if (layerRect.top < 0) {
    layer.style.top = '10px';
  }
}

window.addEventListener('resize', adjustCorrectionImagePosition);
window.addEventListener('scroll', adjustCorrectionImagePosition);

function adjustCorrectionImagePosition() {
  const selectedElement = document.querySelector('.selected-image, video');
  const correctionLayer = document.getElementById('image-layer');

  if (selectedElement && correctionLayer) {
    const rect = selectedElement.getBoundingClientRect();
    correctionLayer.style.top = `${rect.top + window.scrollY}px`;
    correctionLayer.style.left = `${rect.right + window.scrollX + 10}px`;

    // Adjust position if out of viewport
    const layerRect = correctionLayer.getBoundingClientRect();

    // Ensure the layer is within the viewport horizontally
    if (layerRect.right > window.innerWidth) {
      correctionLayer.style.left = `${rect.left + window.scrollX - layerRect.width - 10}px`;
    }

    // Ensure the layer is within the viewport vertically (bottom boundary)
    if (layerRect.bottom > window.innerHeight) {
      correctionLayer.style.top = `${window.innerHeight - layerRect.height - 10 + window.scrollY}px`;
    }

    // Ensure the layer is within the viewport vertically (top boundary)
    if (layerRect.top < 0) {
      correctionLayer.style.top = '10px';
    }
  }
}

// Function to start the capture process
function startCaptureProcess() {
  const video = document.querySelector('video');
  if (!video) {
    alert('No video element found on this page.');
    return;
  }

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64ImageData = canvas.toDataURL('image/jpeg');
  const rect = video.getBoundingClientRect();
  showLoadingNextToElement(rect); // 로딩 화면 표시
  sendImageDataToServer(base64ImageData, rect); // 수정된 부분
}

function showLoadingNextToElement(rect) {
  const layer = document.createElement('div');
  layer.id = 'loading-layer';
  layer.style.position = 'absolute';
  layer.style.top = `${rect.top + window.scrollY}px`; // 클릭한 이미지의 절대 위쪽 위치
  layer.style.left = `${rect.right + window.scrollX + 10}px`; // 클릭한 이미지의 절대 오른쪽 위치
  layer.style.width = `${rect.width}px`; // 클릭한 이미지의 너비
  layer.style.height = `${rect.height}px`; // 클릭한 이미지의 높이
  layer.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
  layer.style.zIndex = '9999';
  layer.style.display = 'flex';
  layer.style.justifyContent = 'center';
  layer.style.alignItems= 'center';
  layer.style.overflow = 'hidden';

  const loadingDiv = document.createElement('div');
  loadingDiv.classList.add('loading');
  
  const span1 = document.createElement('span');
  const span2 = document.createElement('span');
  const span3 = document.createElement('span');

  loadingDiv.appendChild(span1);
  loadingDiv.appendChild(span2);
  loadingDiv.appendChild(span3);

  layer.appendChild(loadingDiv);
  document.body.appendChild(layer);

  // Adjust position if out of viewport
  const layerRect = layer.getBoundingClientRect();

  // Ensure the layer is within the viewport horizontally
  if (layerRect.right > window.innerWidth) {
    layer.style.left = `${rect.left + window.scrollX - layerRect.width - 10}px`;
  }

  // Ensure the layer is within the viewport vertically (bottom boundary)
  if (layerRect.bottom > window.innerHeight) {
    layer.style.top = `${window.innerHeight - layerRect.height - 10 + window.scrollY}px`;
  }

  // Ensure the layer is within the viewport vertically (top boundary)
  if (layerRect.top < 0) {
    layer.style.top = '10px';
  }
}

function showLoadingError() {
  const loadingLayer = document.getElementById('loading-layer');
  if (loadingLayer) {
    loadingLayer.innerHTML = ''; // Clear existing loading animation
    const errorMessage = document.createElement('div');
    errorMessage.innerText = 'Error';
    errorMessage.style.color = 'white';
    errorMessage.style.fontSize = '16px';
    loadingLayer.appendChild(errorMessage);
  }
}

function removeAllLayers() {
  const layers = document.querySelectorAll('#image-layer, #loading-layer');
  layers.forEach(layer => layer.remove());
}
