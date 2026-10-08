import { useState, useEffect } from 'react'
import './index.css'

const DATA_URL = "http://localhost:8000"

const modules = [
  "1. Dataset Loading",
  "2. Preprocessing",
  "3. SAM 2 Segmentation",
  "4. Controlled Perturbation",
  "5. Confidence & Uncertainty",
  "6. Mask Refinement",
  "7. Component Extraction",
  "8. Quantity Analysis",
  "9. Spatial Analysis",
  "10. Logical Anomaly Scoring",
  "11. Final Evaluation"
]

function App() {
  const [activeModule, setActiveModule] = useState(modules[0])
  const [quantityData, setQuantityData] = useState(null)
  const [evalData, setEvalData] = useState(null)

  useEffect(() => {
    fetch(`${DATA_URL}/module8_quantity_features.json`)
      .then(res => res.json())
      .then(data => setQuantityData(data))
      .catch(err => console.error("Could not load quantity JSON", err))
      
    // Assuming we could fetch evaluation CSV but for now we rely on images
  }, [])

  const renderModuleContent = () => {
    switch (activeModule) {
      case "1. Dataset Loading":
        return (
          <div className="card">
            <h2>Dataset Loading</h2>
            <p><strong>Dataset:</strong> MVTec LOCO AD</p>
            <p><strong>Category:</strong> Breakfast Box</p>
            <p><strong>Splits:</strong> Train, Validation, Test</p>
            <div className="image-container">
              <img src={`${DATA_URL}/mvtec_loco_ad/breakfast_box/test/good/000.png`} alt="Sample Dataset Image" onError={(e) => { e.target.style.display = 'none'; e.target.insertAdjacentHTML('afterend', '<p class="error">Image not available</p>')}} />
            </div>
          </div>
        )
      case "2. Preprocessing":
        return (
          <div className="card">
            <h2>Image Preprocessing</h2>
            <p>Module 2 prepares raw images for segmentation by scaling and padding.</p>
            <ul>
              <li><strong>Original Dimensions:</strong> variable</li>
              <li><strong>Target Dimensions:</strong> 1280 x 1600</li>
              <li><strong>Normalization:</strong> 0.0 to 1.0 (float32)</li>
            </ul>
          </div>
        )
      case "3. SAM 2 Segmentation":
        return (
          <div className="card">
            <h2>SAM 2 Segmentation</h2>
            <p><strong>Model:</strong> SAM 2.1 Hiera Tiny (CPU)</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module3_segmentation_result.png`} alt="SAM 2 Result" />
            </div>
          </div>
        )
      case "4. Controlled Perturbation":
        return (
          <div className="card">
            <h2>Controlled Perturbation</h2>
            <p>Introduces erosion, dilation, boundary distortion, and noise across different severity levels.</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module4_perturbation_result.png`} alt="Perturbation Result" />
            </div>
          </div>
        )
      case "5. Confidence & Uncertainty":
        return (
          <div className="card">
            <h2>Confidence & Uncertainty</h2>
            <p>Scores the reliability of the segmentation masks.</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module5_confidence_result.png`} alt="Confidence Result" />
            </div>
          </div>
        )
      case "6. Mask Refinement":
        return (
          <div className="card">
            <h2>Confidence-Aware Mask Refinement</h2>
            <p>Filters out unreliable masks using a defined confidence threshold (e.g., 0.50).</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module6_refinement_result.png`} alt="Refinement Result" />
            </div>
          </div>
        )
      case "7. Component Extraction":
        return (
          <div className="card">
            <h2>Component Extraction</h2>
            <p>Translates pixel masks into localized component properties (Centroids, Areas, BBoxes).</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module7_component_extraction_result.png`} alt="Component Extraction" />
            </div>
          </div>
        )
      case "8. Quantity Analysis":
        return (
          <div className="card">
            <h2>Quantity-Based Feature Analysis</h2>
            <p>Calculates the number of valid components passing refinement.</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module8_quantity_analysis_result.png`} alt="Quantity Analysis" />
            </div>
            {quantityData && (
              <div className="data-table">
                <h3>Data Preview (Severity: 20%)</h3>
                <pre>{JSON.stringify(quantityData["20"], null, 2)}</pre>
              </div>
            )}
          </div>
        )
      case "9. Spatial Analysis":
        return (
          <div className="card">
            <h2>Spatial Relationship Analysis</h2>
            <p>Calculates spatial arrangements (distances, ABOVE/BELOW, NEAR/FAR) to detect logical inconsistencies.</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module9_spatial_relationship_result.png`} alt="Spatial Analysis" />
            </div>
          </div>
        )
      case "10. Logical Anomaly Scoring":
        return (
          <div className="card">
            <h2>Logical Anomaly Scoring</h2>
            <p>Fuses spatial and quantity deviations to assign a final logical anomaly score (0=Normal, 1=Anomalous).</p>
            <div className="image-container">
              <img src={`${DATA_URL}/module10_logical_anomaly_result.png`} alt="Logical Anomaly Scoring" />
            </div>
          </div>
        )
      case "11. Final Evaluation":
        return (
          <div className="card">
            <h2>Final Evaluation</h2>
            <p>Demonstrates the massive robustness gain of the proposed confidence-aware pipeline against severe degradation.</p>
            <div className="image-container">
              <img src={`${DATA_URL}/evaluation_results/robustness_evaluation.png`} alt="Robustness Evaluation" />
            </div>
            <p>A full CSV of the evaluation is available in the data directory.</p>
          </div>
        )
      default:
        return <div>Select a module</div>
    }
  }

  return (
    <div className="dashboard-container">
      <nav className="sidebar">
        <h1 className="logo">Robust AD</h1>
        <ul className="nav-list">
          {modules.map((mod) => (
            <li 
              key={mod} 
              className={activeModule === mod ? 'active' : ''}
              onClick={() => setActiveModule(mod)}
            >
              {mod}
            </li>
          ))}
        </ul>
      </nav>
      <main className="content">
        <header className="topbar">
          <h2>Enhanced Systematic Robustness Analysis for Product Image Anomaly Detection</h2>
        </header>
        <div className="module-content">
          {renderModuleContent()}
        </div>
      </main>
    </div>
  )
}

export default App
