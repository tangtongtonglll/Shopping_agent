/**
 * API Client for Browser Extension
 * Client for communicating with backend API
 */

const API_BASE_URL = 'http://localhost:8000';

class APIClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getConfig() {
    const result = await chrome.storage.sync.get(['config']);
    return result.config || { apiUrl: this.baseUrl };
  }

  async request(endpoint, options = {}) {
    const config = await this.getConfig();
    const url = `${config.apiUrl || this.baseUrl}${endpoint}`;
    
    const defaultOptions = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    };
    
    const mergedOptions = {
      ...defaultOptions,
      ...options,
      headers: {
        ...defaultOptions.headers,
        ...options.headers,
      },
    };
    
    // If FormData, don't set Content-Type, let browser set it automatically
    if (mergedOptions.body instanceof FormData) {
      delete mergedOptions.headers['Content-Type'];
    } else if (mergedOptions.body && typeof mergedOptions.body === 'object') {
      mergedOptions.body = JSON.stringify(mergedOptions.body);
    }
    
    try {
      const response = await fetch(url, mergedOptions);
      
      if (!response.ok) {
        // Try to get detailed error information
        let errorMessage = `API request failed: ${response.statusText} (${response.status})`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          // If unable to parse JSON, try to read text
          try {
            const errorText = await response.text();
            if (errorText) {
              errorMessage = errorText.substring(0, 200);
            }
          } catch (e2) {
            // Ignore parsing errors
          }
        }
        throw new Error(errorMessage);
      }
      
      // Check response content type
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        // If not JSON, return text
        const text = await response.text();
        try {
          return JSON.parse(text);
        } catch (e) {
          return { message: text };
        }
      }
    } catch (error) {
      console.error('API request error:', error);
      // Provide more user-friendly error information
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        throw new Error('Failed to connect to server. Please ensure the backend service is running at http://localhost:8000');
      }
      throw error;
    }
  }

  // Chat-related APIs
  async sendChatMessage(message, conversationId = null, options = {}) {
    return this.request('/api/chat/chat', {
      method: 'POST',
      body: {
        message,
        conversation_id: conversationId,
        message_type: options.message_type || 'text',
        model: options.model || 'glm-4-0520',
        use_memory: options.use_memory !== undefined ? options.use_memory : true,  // Enable memory by default
        use_rag: options.use_rag !== undefined ? options.use_rag : false,
        max_tokens: options.max_tokens,
        temperature: options.temperature,
      },
    });
  }

  async sendEnhancedChat(request) {
    return this.request('/api/chat/enhanced', {
      method: 'POST',
      body: request,
    });
  }

  // Shopping-related APIs
  async searchProduct(query, platforms = []) {
    return this.request('/api/shopping/search', {
      method: 'POST',
      body: {
        query,
        platforms,
      },
    });
  }

  async analyzeProduct(productData) {
    return this.request('/api/shopping/product-analysis', {
      method: 'POST',
      body: productData,
    });
  }

  async comparePrices(productName, platforms = ['jd', 'taobao', 'pdd']) {
    // Use shopping API's price comparison feature, which uses database data
    return this.request('/api/shopping/price-comparison', {
      method: 'POST',
      body: {
        query: productName,
        platforms: platforms.map(p => {
          // Convert platform name format
          const platformMap = {
            'jd': 'jd',
            'taobao': 'taobao',
            'pdd': 'pdd',
            'jd.com': 'jd',
            'taobao.com': 'taobao',
            'pdd.com': 'pdd'
          };
          return platformMap[p.toLowerCase()] || p;
        }),
      },
    });
  }

  async trackPrice(productId, targetPrice = null) {
    return this.request('/api/price-tracker/track', {
      method: 'POST',
      body: {
        product_id: productId,
        target_price: targetPrice,
      },
    });
  }

  async getPriceHistory(productId, days = 30) {
    return this.request(`/api/price-tracker/analysis/${productId}?days=${days}`);
  }

  async predictPrice(productId) {
    return this.request(`/api/ecommerce/products/${productId}/price-prediction`);
  }

  async analyzeRisk(productId) {
    return this.request(`/api/ecommerce/products/${productId}/risk-analysis`);
  }

  // Product recommendations
  async getRecommendations(userId, options = {}) {
    return this.request('/api/comparison/recommendations', {
      method: 'POST',
      body: {
        user_id: userId,
        ...options,
      },
    });
  }

  // Visual search（将文件转为 base64 后调用 /search/base64 接口）
  async visualSearch(imageFile) {
    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // 去掉 "data:image/xxx;base64," 前缀
        const b64 = reader.result.split(',')[1];
        resolve(b64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(imageFile);
    });

    return this.request('/api/visual-search/search/base64', {
      method: 'POST',
      body: { image_data: base64, max_results: 10 },
    });
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }
}

// Export API client instance
const apiClient = new APIClient();

// If in browser environment, mount it to window
if (typeof window !== 'undefined') {
  window.apiClient = apiClient;
}

// If in CommonJS environment
if (typeof module !== 'undefined' && module.exports) {
  module.exports = apiClient;
}

