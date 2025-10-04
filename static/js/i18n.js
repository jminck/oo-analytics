/**
 * Internationalization (i18n) module for Portfolio Strategy Analytics
 * Supports dynamic language switching and string translation
 */
class I18n {
    constructor() {
        this.currentLanguage = 'en';
        this.translations = {};
        this.loadLanguage();
    }

    /**
     * Load translations for the specified language
     * @param {string} lang - Language code (e.g., 'en', 'es')
     */
    async loadLanguage(lang = null) {
        if (lang) this.currentLanguage = lang;
        
        try {
            const response = await fetch(`/translations/${this.currentLanguage}/frontend.json`);
            if (!response.ok) {
                throw new Error(`Failed to load translations: ${response.status}`);
            }
            this.translations = await response.json();
            this.updateUI();
            this.updateLanguageSwitcher();
        } catch (error) {
            console.error('Failed to load translations:', error);
            // Fallback to English if current language fails
            if (this.currentLanguage !== 'en') {
                this.currentLanguage = 'en';
                await this.loadLanguage();
            }
        }
    }

    /**
     * Translate a string with optional parameters
     * @param {string} key - Translation key
     * @param {Object} params - Parameters to replace in the translation
     * @returns {string} Translated string
     */
    t(key, params = {}) {
        let translation = this.translations[key] || key;
        
        // Debug chart translations
        if (key.startsWith('chart_')) {
            console.log('🔧 Translation request for:', key, '->', translation);
        }
        
        // Replace parameters in the format {{param}}
        Object.keys(params).forEach(param => {
            translation = translation.replace(new RegExp(`{{${param}}}`, 'g'), params[param]);
        });
        
        return translation;
    }

    /**
     * Update all UI elements with data-i18n attributes
     */
    updateUI() {
        console.log('🔄 updateUI called, current language:', this.currentLanguage);
        console.log('🔄 Available translations:', Object.keys(this.translations).length);
        
        // Update elements with data-i18n attributes
        const i18nElements = document.querySelectorAll('[data-i18n]');
        console.log('🔄 Found i18n elements:', i18nElements.length);
        
        i18nElements.forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            // Debug Live vs Backtest table headers
            if (key.startsWith('live_vs_bt_')) {
                console.log('🔄 Live vs BT element:', key, '->', translation);
            }
            
            if (element.tagName === 'INPUT' && (element.type === 'text' || element.type === 'email' || element.type === 'password')) {
                element.placeholder = translation;
            } else if (element.hasAttribute('data-i18n-placeholder')) {
                element.placeholder = translation;
            } else if (element.tagName === 'TITLE') {
                element.textContent = translation;
            } else {
                element.textContent = translation;
            }
        });

        // Update title attributes
        document.querySelectorAll('[data-i18n-title]').forEach(element => {
            const key = element.getAttribute('data-i18n-title');
            element.title = this.t(key);
        });

        // Update tooltip content
        document.querySelectorAll('[data-i18n-tooltip]').forEach(element => {
            const key = element.getAttribute('data-i18n-tooltip');
            element.setAttribute('data-tooltip', this.t(key));
        });

        // Update chart titles and descriptions
        this.updateChartTitles();
        
        // Refresh chart buttons with new translations
        setTimeout(() => {
            if (typeof renderChartButtons === 'function') {
                console.log('🔄 Refreshing chart buttons with new translations...');
                console.log('🔄 Current language:', this.currentLanguage);
                console.log('🔄 Available translations:', Object.keys(this.translations).length);
                console.log('🔄 Testing chart_overview translation:', this.t('chart_overview'));
                
                // Ensure i18n is ready before rendering
                if (Object.keys(this.translations).length > 0) {
                    renderChartButtons();
                } else {
                    console.log('⚠️ i18n translations not ready, retrying in 200ms...');
                    setTimeout(() => {
                        if (Object.keys(this.translations).length > 0) {
                            renderChartButtons();
                        } else {
                            console.log('❌ i18n translations still not ready after retry');
                        }
                    }, 200);
                }
            } else {
                console.log('❌ renderChartButtons function not found');
            }
            
            // Re-render Live vs Backtest table if it exists and has data
            if (typeof renderLiveVsBtTable === 'function' && window.liveVsBtTableData && window.liveVsBtTableData.length > 0) {
                console.log('🔄 Re-rendering Live vs Backtest table with new translations...');
                console.log('🔄 Live vs BT table data length:', window.liveVsBtTableData.length);
                renderLiveVsBtTable();
            } else {
                console.log('🔄 Live vs Backtest table not ready for re-render');
                console.log('🔄 renderLiveVsBtTable function exists:', typeof renderLiveVsBtTable === 'function');
                console.log('🔄 liveVsBtTableData exists:', !!window.liveVsBtTableData);
                console.log('🔄 liveVsBtTableData length:', window.liveVsBtTableData ? window.liveVsBtTableData.length : 'undefined');
            }
        }, 100);
    }

    /**
     * Update chart titles and descriptions
     */
    updateChartTitles() {
        // Update chart button text
        const chartButtons = document.querySelectorAll('#chartButtons .btn-chart');
        chartButtons.forEach(button => {
            const chartType = button.getAttribute('data-chart-type');
            if (chartType && this.translations[`chart_${chartType}`]) {
                button.textContent = this.t(`chart_${chartType}`);
            }
        });

        // Update section titles
        const sectionTitles = document.querySelectorAll('.section-title');
        sectionTitles.forEach(title => {
            const key = title.getAttribute('data-i18n');
            if (key) {
                title.textContent = this.t(key);
            }
        });
    }

    /**
     * Update language switcher display
     */
    updateLanguageSwitcher() {
        const currentLangElement = document.getElementById('currentLanguage');
        if (currentLangElement) {
            currentLangElement.textContent = this.t(`language_${this.currentLanguage}`);
        }
    }

    /**
     * Switch to a different language
     * @param {string} lang - Language code to switch to
     */
    async switchLanguage(lang) {
        if (lang === this.currentLanguage) return;
        
        try {
            // Update backend language
            const response = await fetch(`/api/set-language/${lang}`);
            if (!response.ok) {
                throw new Error('Failed to set language on server');
            }
            
            // Load new translations
            await this.loadLanguage(lang);
            
            // Show success message
            this.showLanguageChangeMessage(lang);
            
        } catch (error) {
            console.error('Failed to switch language:', error);
            this.showLanguageChangeMessage(lang, false);
        }
    }

    /**
     * Show language change confirmation message
     * @param {string} lang - Language code
     * @param {boolean} success - Whether the change was successful
     */
    showLanguageChangeMessage(lang, success = true) {
        const message = success 
            ? this.t('language_changed_success', { lang: this.t(`language_${lang}`) })
            : this.t('language_changed_error');
        
        // Create temporary alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${success ? 'success' : 'danger'} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 3000);
    }

    /**
     * Get current language
     * @returns {string} Current language code
     */
    getCurrentLanguage() {
        return this.currentLanguage;
    }

    /**
     * Get available languages
     * @returns {Object} Available languages with their display names
     */
    getAvailableLanguages() {
        return {
            'en': this.t('language_en'),
            'es': this.t('language_es')
        };
    }
}

// Create global instance
window.i18n = new I18n();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Language switcher event listeners
    document.querySelectorAll('[data-lang-switch]').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const lang = this.getAttribute('data-lang-switch');
            window.i18n.switchLanguage(lang);
        });
    });
});
