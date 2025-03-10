#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <dirent.h>
#include <cstdint>
#include <iomanip>

void processFile(const std::string& inputFilePath) {
    std::string outputFilePath = inputFilePath.substr(0, inputFilePath.find_last_of('.')) + ".h";

    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Could not open file: " << inputFilePath << std::endl;
        return;
    }

    std::ofstream outputFile(outputFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Could not create output file: " << outputFilePath << std::endl;
        return;
    }

    std::string line;
    std::vector<uint64_t> result;
    result.reserve(4096); // Pre-allocate space for 4096 uint64_t elements

    int lineCount = 0;
    while (std::getline(inputFile, line)) {
        // Trim any whitespace
        line.erase(0, line.find_first_not_of(' '));
        line.erase(line.find_last_not_of(' ') + 1);

        if (line.empty()) continue;

        lineCount++;

        __uint128_t num128 = 0;
        bool valid = true;

        for (char c : line) {
            if (c < '0' || c > '9') {
                std::cerr << "Invalid character in line: " << line << std::endl;
                valid = false;
                break;
            }
            num128 = num128 * 10 + (c - '0');
        }

        if (!valid) continue;

        uint64_t low = static_cast<uint64_t>(num128 & 0xFFFFFFFFFFFFFFFF);
        uint64_t high = static_cast<uint64_t>(num128 >> 64);

        // Debug output to verify conversion
        std::cout << "Line " << lineCount << ": " << line << "\n";
        std::cout << "  High 64 bits: " << high << "\n";
        std::cout << "  Low 64 bits:  " << low << "\n";

        result.push_back(low);
        result.push_back(high);
    }

    // Check for padding
    while (result.size() < 4096) {
        result.push_back(0);
    }

    if (result.size() != 4096) {
        std::cerr << "Warning: File " << inputFilePath << " did not produce exactly 4096 uint64_t elements." << std::endl;
    }

    std::string arrayName = inputFilePath.substr(inputFilePath.find_last_of('/') + 1);
    arrayName = arrayName.substr(0, arrayName.find_last_of('.'));

    outputFile << "#ifndef " << arrayName << "_H\n";
    outputFile << "#define " << arrayName << "_H\n\n";
    outputFile << "#include <cstdint>\n\n";
    outputFile << "uint64_t " << arrayName << "[4096] = {";

    for (size_t i = 0; i < result.size(); ++i) {
        outputFile << result[i];
        if (i < result.size() - 1) outputFile << ", ";
        if ((i + 1) % 8 == 0) outputFile << "\n";
    }

    outputFile << "};\n\n";
    outputFile << "#endif // " << arrayName << "_H\n";

    inputFile.close();
    outputFile.close();
    std::cout << "Header file saved to: " << outputFilePath << std::endl;
}

int main() {
    std::string inputDir = "./input_files"; // Replace with your actual input directory path

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(inputDir.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string filename = ent->d_name;
            if (filename == "." || filename == "..") continue;

            std::string filePath = inputDir + "/" + filename;
            processFile(filePath);
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open input directory: " << inputDir << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
