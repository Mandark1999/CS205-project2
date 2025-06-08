#include <iostream>    
#include <fstream>    
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdlib>
using namespace std;

struct Instance {
    int label;
    vector<double> features;
};

vector<Instance> loadAndNormalize(const string& filename, int& num_features) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: Cannot open file " << filename << "\n";
        exit(1);
    }
    vector<vector<double>> raw_data;
    vector<int> labels;
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        double val;
        ss >> val;
        labels.push_back(static_cast<int>(val));
        vector<double> row;
        while (ss >> val) {
            row.push_back(val);
        }
        raw_data.push_back(row);
    }
    fin.close();

    int num_instances = raw_data.size();
    if (num_instances == 0) {
        cerr << "Error: Empty dataset\n";
        exit(1);
    }
    num_features = raw_data[0].size();

    vector<double> mean_features(num_features, 0.0), std_features(num_features, 0.0);
    for (int j = 0; j < num_features; j++) {
        for (int i = 0; i < num_instances; i++) {
            mean_features[j] += raw_data[i][j];
        }
        mean_features[j] /= num_instances;
        for (int i = 0; i < num_instances; i++) {
            double diff = raw_data[i][j] - mean_features[j];
            std_features[j] += diff * diff;
        }
        std_features[j] = sqrt(std_features[j] / (num_instances - 1));
        if (std_features[j] == 0.0) std_features[j] = 1.0;
    }

    vector<Instance> instances(num_instances);
    for (int i = 0; i < num_instances; i++) {
        instances[i].label = labels[i];
        instances[i].features.resize(num_features);
        for (int j = 0; j < num_features; j++) {
            instances[i].features[j] = (raw_data[i][j] - mean_features[j]) / std_features[j];
        }
    }
    return instances;
}

double evaluateSubset(const vector<Instance>& instances, const vector<int>& subset) {
    int num_instances = instances.size();
    int correct = 0;
    for (int i = 0; i < num_instances; i++) {
        const Instance& test_instance = instances[i];
        double best_distance = numeric_limits<double>::max();
        int best_label = -1;
        for (int j = 0; j < num_instances; j++) {
            if (j == i) continue;
            const Instance& candidate = instances[j];
            double d2 = 0.0;
            for (int fIdx : subset) {
                double diff = test_instance.features[fIdx] - candidate.features[fIdx];
                d2 += diff * diff;
            }
            if (d2 < best_distance) {
                best_distance = d2;
                best_label = candidate.label;
            }
        }
        if (best_label == test_instance.label) correct++;
    }
    return static_cast<double>(correct) / num_instances;
}

vector<int> forwardSelection(const vector<Instance>& instances, int num_features) {
    vector<int> current_set;
    double best_accuracy = evaluateSubset(instances, current_set);
    int count_class1 = 0, num_instances = instances.size();
    for (const auto& inst : instances) {
        if (inst.label == 1) count_class1++;
    }
    int majorityCount = max(count_class1, num_instances - count_class1);
    double defaultRate = static_cast<double>(majorityCount) / num_instances;
    cout << "Default rate (no features): " << defaultRate * 100 << "%\n";

    vector<bool> feature_used(num_features, false);
    vector<int> bestSubset = current_set;

    for (int level = 1; level <= num_features; level++) {
        double level_best_accuracy = 0.0;
        int level_best_feature = -1;
        for (int f = 0; f < num_features; f++) {
            if (feature_used[f]) continue;
            vector<int> trial_set = current_set;
            trial_set.push_back(f);
            double acc = evaluateSubset(instances, trial_set);
            cout << "Using feature(s) {";
            for (int idx = 0; idx < trial_set.size(); idx++) {
                cout << trial_set[idx] + 1;
                if (idx < trial_set.size() - 1) cout << ",";
            }
            cout << "} accuracy is " << acc * 100 << "%\n";

            if (acc > level_best_accuracy) {
                level_best_accuracy = acc;
                level_best_feature = f;
            }
        }
        if (level_best_feature < 0 || level_best_accuracy <= best_accuracy) {
            break;
        }
        feature_used[level_best_feature] = true;
        current_set.push_back(level_best_feature);
        best_accuracy = level_best_accuracy;
        bestSubset = current_set;
        cout << "→ Feature set {";
        for (int i = 0; i < current_set.size(); i++) {
            cout << current_set[i] + 1;
            if (i < current_set.size() - 1) cout << ",";
        }
        cout << "} was best, accuracy is " << best_accuracy * 100 << "%\n";
    }
    cout << "Finished search!! The best feature subset is {";
    for (int i = 0; i < bestSubset.size(); i++) {
        cout << bestSubset[i] + 1;
        if (i < bestSubset.size() - 1) cout << ",";
    }
    cout << "}, which has an accuracy of " << best_accuracy * 100 << "%\n";

    return bestSubset;
}

vector<int> backwardElimination(const vector<Instance>& data, int F) {
    vector<int> currentSet(F);
    iota(currentSet.begin(), currentSet.end(), 0);
    double bestSoFar = evaluateSubset(data, currentSet);
    cout << "Baseline (all features): " << bestSoFar * 100 << "%\n";

    vector<bool> used(F, true);
    vector<int> bestSubset = currentSet;

    for (int level = F; level >= 1; level--) {
        double levelBestAcc = 0.0;
        int levelRemoveFeature = -1;
        vector<int> levelBestSet;
        for (int f = 0; f < F; f++) {
            if (!used[f]) continue;
            vector<int> trialSet;
            for (int x : currentSet) {
                if (x != f) trialSet.push_back(x);
            }
            double acc = evaluateSubset(data, trialSet);
            cout << "Using feature(s) {";
            for (int idx = 0; idx < trialSet.size(); idx++) {
                cout << trialSet[idx] + 1;
                if (idx < trialSet.size() - 1) cout << ",";
            }
            cout << "} accuracy is " << acc * 100 << "%\n";

            if (acc > levelBestAcc) {
                levelBestAcc = acc;
                levelRemoveFeature = f;
                levelBestSet = trialSet;
            }
        }
        if (levelRemoveFeature < 0 || levelBestAcc <= bestSoFar) {
            break;
        }
        used[levelRemoveFeature] = false;
        currentSet = levelBestSet;
        bestSoFar = levelBestAcc;
        bestSubset = currentSet;
        cout << "→ Feature set {";
        for (int i = 0; i < currentSet.size(); i++) {
            cout << currentSet[i] + 1;
            if (i < currentSet.size() - 1) cout << ",";
        }
        cout << "} was best, accuracy is " << bestSoFar * 100 << "%\n";
    }
    cout << "Finished search!! The best feature subset is {";
    for (int i = 0; i < bestSubset.size(); i++) {
        cout << bestSubset[i] + 1;
        if (i < bestSubset.size() - 1) cout << ",";
    }
    cout << "}, which has an accuracy of " << bestSoFar * 100 << "%\n";

    return bestSubset;
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "Welcome to the Feature Selection Algorithm.\n";
    cout << "Type in the name of the file to test: ";
    string filename;
    cin >> filename;

    cout << "Type the number of the algorithm you want to run.\n"
         << "  1) Forward Selection\n"
         << "  2) Backward Elimination\n";
    int choice;
    cin >> choice;

    int num_features;
    vector<Instance> instances = loadAndNormalize(filename, num_features);
    int num_instances = instances.size();
    cout << "Dataset has " << num_features << " features, " << num_instances << " instances.\n\n";

    if (choice == 1) {
        forwardSelection(instances, num_features);
    } else if (choice == 2) {
        backwardElimination(instances, num_features);
    } else {
        cout << "Invalid choice. Exiting.\n";
    }

    return 0;
}