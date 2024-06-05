#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;
int main() {
  long long inf = 1e18;
  int N;
  cin >> N;
  vector<int> X(N), Y(N), Z(N);
  for (int i = 0; i < N; i++) cin >> X[i] >> Y[i] >> Z[i];
  int z_sum = accumulate(begin(Z), end(Z), 0);
  vector<long long> dp(z_sum + 1, inf);
  dp[0] = 0;
  for (int i = 0; i < N; i++) {
    int x = X[i], y = Y[i], z = Z[i];
    int w = max(0, (x + y) / 2 + 1 - x);
    for (int j = z_sum; j >= z; j--) {
      dp[j] = min(dp[j], dp[j - z] + w);
    }
  }
  long long ans = inf;
  for (int j = z_sum / 2 + 1; j <= z_sum; j++) ans = min(ans, dp[j]);
  cout << ans << endl;
}