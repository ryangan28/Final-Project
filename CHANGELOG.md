# Changelog

All notable changes to the Organic Farm Pest Management AI System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-07-29

### Added
- **Split dependency system**: Separate `requirements-demo.txt` and `requirements-full.txt` for lightweight vs production deployment
- **Comprehensive documentation**: Added `docs/evaluation.md` with detailed performance metrics and validation methodology
- **Installation guide**: Added `docs/installation.md` with step-by-step setup instructions
- **Internationalization foundation**: Created `locales/en.json` for future multi-language support
- **Enhanced system status**: Added dependency checking and optimization warnings in web interface
- **Security improvements**: Implemented secure temporary file handling with automatic cleanup
- **Docker support**: Added Dockerfile examples for containerized deployment

### Changed
- **Updated README**: Added performance metrics, documentation links, and improved quick start guide
- **Enhanced error handling**: Better graceful degradation when ML dependencies are missing
- **Improved system status display**: Shows specific missing dependencies and their impact

### Security
- **Temporary file handling**: Replaced insecure temp file creation with `tempfile.NamedTemporaryFile()`
- **Automatic cleanup**: Added proper file cleanup with try/finally blocks

### Performance
- **Edge optimization**: Enhanced model optimization with better fallback handling
- **Memory efficiency**: Improved resource management for edge deployment

## [2.0.0] - 2025-07-28

### Added
- **Complete treatment recommendation engine**: 8 pest types with severity-aware organic treatments
- **Vision module enhancement**: Added PyTorch tensor support with graceful fallbacks
- **Chat interface improvements**: Fixed greeting detection with regex word boundaries
- **System integration**: Full module-level imports for test compatibility
- **Edge optimization**: Added dependency fallbacks for missing psutil/onnx libraries

### Changed
- **Test success rate**: Achieved 100% test coverage (27/27 tests passing)
- **Organic compliance**: All treatment recommendations now OMRI-certified
- **IPM principles**: Integrated Integrated Pest Management throughout system

### Fixed
- **Treatment engine**: Resolved empty database causing 13 test failures
- **Vision module**: Fixed tensor interface compatibility issues
- **Chat detection**: Resolved false positive greeting detection
- **File permissions**: Fixed Windows file permission errors in tests

## [1.0.0] - 2025-07-27

### Added
- Initial release of Organic Farm Pest Management AI System
- **Computer vision pest detection**: 8 common agricultural pests
- **Conversational AI assistant**: Natural language interface for farmers
- **Organic treatment database**: OMRI-approved treatments only
- **Mobile-friendly web interface**: Streamlit-based application
- **Edge computing optimization**: Offline-capable deployment
- **Comprehensive test suite**: Unit and integration tests

### Features
- **Pest identification**: Aphids, Caterpillars, Spider Mites, Whitefly, Thrips, Colorado Potato Beetle, Cucumber Beetle, Flea Beetle
- **Treatment categories**: Biological, Cultural, Mechanical, and Preventive controls
- **IPM principles**: Integrated approach emphasizing prevention
- **Offline operation**: 100% functionality without internet connectivity
- **Modular architecture**: Separate vision, treatment, chat, and edge modules

---

## Versioning Strategy

- **Major version** (X.0.0): Breaking changes, new core features
- **Minor version** (1.X.0): New features, enhancements, significant improvements
- **Patch version** (1.1.X): Bug fixes, security patches, minor improvements

## Development Guidelines

- All changes must maintain 100% test coverage
- New features require documentation updates
- Security improvements are prioritized
- Backward compatibility maintained within major versions
