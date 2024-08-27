import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { PCA } from 'ml-pca';

const EmbeddingVisualizer = () => {
  const [strings, setStrings] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [embeddings, setEmbeddings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedString, setSelectedString] = useState('');
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const pointsRef = useRef(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  const selectedIndexRef = useRef(-1);
  const hoveredIndexRef = useRef(-1);
  const uniformsRef = useRef(null);

  const getEmbeddings = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          input: strings.split('\n'),
          model: 'text-embedding-ada-002'
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch embeddings');
      }

      const data = await response.json();
      const newEmbeddings = data.data.map(item => item.embedding);
      console.log('Fetched embeddings:', newEmbeddings);
      setEmbeddings(newEmbeddings);
    } catch (err) {
      setError('Error fetching embeddings: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('useEffect triggered. Embeddings length:', embeddings.length);
    if (embeddings.length > 0 && canvasRef.current) {
      console.log('Creating scene');
      const scene = new THREE.Scene();
      sceneRef.current = scene;
      scene.background = new THREE.Color(0xf0f0f0);  // Light gray background

      const aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
      const camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
      cameraRef.current = camera;
      const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true });
      renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);

      const controls = new OrbitControls(camera, renderer.domElement);

      // Add AxesHelper
      const axesHelper = new THREE.AxesHelper(5);
      scene.add(axesHelper);

      // Add axes labels
      const addAxisLabel = (text, position) => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = 'Bold 60px Arial';
        context.fillStyle = 'black';
        context.fillText(text, 0, 60);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(0.5, 0.5, 0.5);
        scene.add(sprite);
      };

      addAxisLabel('X', new THREE.Vector3(5.2, 0, 0));
      addAxisLabel('Y', new THREE.Vector3(0, 5.2, 0));
      addAxisLabel('Z', new THREE.Vector3(0, 0, 5.2));

      // Perform PCA to reduce dimensionality to 3D
      console.log('Performing PCA');
      const pca = new PCA(embeddings);
      const reducedData = pca.predict(embeddings, { nComponents: 3 });
      console.log('Reduced data sample:', reducedData.getRow(0), reducedData.getRow(1));

      // Extract and normalize the positions
      let minVal = Infinity;
      let maxVal = -Infinity;
      const positions = new Float32Array(reducedData.rows * 3);
      for (let i = 0; i < reducedData.rows; i++) {
        const row = reducedData.getRow(i);
        for (let j = 0; j < 3; j++) {
          const val = row[j];
          positions[i * 3 + j] = val;
          minVal = Math.min(minVal, val);
          maxVal = Math.max(maxVal, val);
        }
      }

      console.log('Min value:', minVal, 'Max value:', maxVal);
      console.log('Positions sample:', positions.slice(0, 9));

      // Normalize and center the points
      const scale = 2 / (maxVal - minVal);
      for (let i = 0; i < positions.length; i++) {
        positions[i] = (positions[i] - minVal) * scale - 1;
      }

      console.log('Normalized positions sample:', positions.slice(0, 9));

      // Create points with custom shaders for better-looking nodes
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      const vertexShader = `
        uniform float selectedIndex;
        uniform float hoveredIndex;
        uniform float time;

        varying vec3 vColor;
        varying float vHighlight;

        void main() {
          vColor = vec3(0.5 + position.x / 2.0, 0.5 + position.y / 2.0, 0.5 + position.z / 2.0);
          
          float isSelected = step(abs(float(gl_VertexID) - selectedIndex), 0.1);
          float isHovered = step(abs(float(gl_VertexID) - hoveredIndex), 0.1);
          
          vHighlight = max(isSelected, isHovered);
          
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          gl_PointSize = 8.0 + vHighlight * 4.0 * (1.0 + 0.2 * sin(time * 5.0));
        }
      `;

      const fragmentShader = `
        uniform float time;
        varying vec3 vColor;
        varying float vHighlight;

        void main() {
          float r = length(gl_PointCoord - vec2(0.5, 0.5));
          if (r > 0.5) discard;
          
          vec3 color = vColor;
          if (vHighlight > 0.5) {
            color = mix(color, vec3(1.0, 1.0, 0.0), 0.5 + 0.5 * sin(time * 5.0));
          }
          
          gl_FragColor = vec4(color, 1.0 - smoothstep(0.3, 0.5, r));
        }
      `;

      const uniforms = {
        time: { value: 0 },
        selectedIndex: { value: -1 },
        hoveredIndex: { value: -1 }
      };
      uniformsRef.current = uniforms;

      const material = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: uniforms,
        transparent: true
      });

      const points = new THREE.Points(geometry, material);
      pointsRef.current = points;
      scene.add(points);

      console.log('Points added to scene:', points);

      // Add ambient light
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambientLight);

      // Add directional light
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
      directionalLight.position.set(1, 1, 1);
      scene.add(directionalLight);

      camera.position.set(5, 5, 5);
      controls.update();

      console.log('Starting animation');
      const animate = () => {
        requestAnimationFrame(animate);
        controls.update();
        uniformsRef.current.time.value += 0.016; // Approximately 60 FPS
        renderer.render(scene, camera);
      };

      animate();

      const handleResize = () => {
        if (canvasRef.current) {
          const newAspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
          camera.aspect = newAspect;
          camera.updateProjectionMatrix();
          renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
        }
      };

      const handleMouseMove = (event) => {
        updateMousePosition(event);
        updateHoveredString();
      };

      const canvas = canvasRef.current;
      canvas.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        if (canvas) {
          canvas.removeEventListener('mousemove', handleMouseMove);
        }
        renderer.dispose();
        controls.dispose();
      };
    }
  }, [embeddings]);

  const updateMousePosition = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  };

  const updateHoveredString = () => {
    if (sceneRef.current && cameraRef.current && pointsRef.current) {
      raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
      const intersects = raycasterRef.current.intersectObject(pointsRef.current);

      if (intersects.length > 0) {
        const index = intersects[0].index;
        const hoveredString = strings.split('\n')[index];
        setSelectedString(hoveredString);
        hoveredIndexRef.current = index;
        uniformsRef.current.hoveredIndex.value = index;
      } else {
        setSelectedString('');
        hoveredIndexRef.current = -1;
        uniformsRef.current.hoveredIndex.value = -1;
      }
    }
  };

  useEffect(() => {
    const handleClick = (event) => {
      updateMousePosition(event);
      if (sceneRef.current && cameraRef.current && pointsRef.current) {
        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
        const intersects = raycasterRef.current.intersectObject(pointsRef.current);

        if (intersects.length > 0) {
          const index = intersects[0].index;
          selectedIndexRef.current = index;
          uniformsRef.current.selectedIndex.value = index;
        } else {
          selectedIndexRef.current = -1;
          uniformsRef.current.selectedIndex.value = -1;
        }
      }
    };

    window.addEventListener('click', handleClick);

    return () => {
      window.removeEventListener('click', handleClick);
    };
  }, [strings]);

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'Arial, sans-serif', backgroundColor: '#f0f0f0' }}>
      <div style={{ width: '30%', padding: '2rem', borderRight: '1px solid #ccc', backgroundColor: '#ffffff', boxShadow: '2px 0 5px rgba(0,0,0,0.1)' }}>
        <h2 style={{ marginTop: 0, marginBottom: '1rem', color: '#333' }}>Embedding Visualizer</h2>
        <textarea
          value={strings}
          onChange={(e) => setStrings(e.target.value)}
          placeholder="Enter strings (one per line)"
          style={{ 
            width: '100%', 
            height: '200px', 
            marginBottom: '1rem', 
            padding: '0.5rem',
            resize: 'vertical',
            border: '1px solid #ccc',
            borderRadius: '4px',
            fontSize: '14px'
          }}
        />
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Enter OpenAI API Key"
          style={{ 
            display: 'block', 
            width: '100%', 
            marginBottom: '1rem', 
            padding: '0.5rem',
            border: '1px solid #ccc',
            borderRadius: '4px',
            fontSize: '14px'
          }}
        />
        <button 
          onClick={getEmbeddings} 
          disabled={loading}
          style={{ 
            padding: '0.75rem 1rem', 
            backgroundColor: '#4CAF50', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer',
            width: '100%',
            fontSize: '16px',
            fontWeight: 'bold',
            transition: 'background-color 0.3s'
          }}
        >
          {loading ? 'Loading...' : 'Generate Embeddings'}
        </button>
        {error && <div style={{ color: 'red', marginTop: '1rem', fontSize: '14px' }}>{error}</div>}
        {selectedString && (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e8f5e9', borderRadius: '4px', border: '1px solid #81c784' }}>
            <strong style={{ display: 'block', marginBottom: '0.5rem', color: '#2e7d32' }}>Selected/Hovered String:</strong>
            <span style={{ fontSize: '14px', color: '#333' }}>{selectedString}</span>
          </div>
        )}
      </div>
      <div style={{ width: '70%', height: '100%', position: 'relative' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
        <div style={{ position: 'absolute', bottom: '1rem', right: '1rem', background: 'rgba(255,255,255,0.7)', padding: '0.5rem', borderRadius: '4px', fontSize: '12px' }}>
          Tip: Use mouse to rotate, scroll to zoom
        </div>
      </div>
    </div>
  );
};

export default EmbeddingVisualizer;