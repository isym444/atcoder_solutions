#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <queue>
#include <array>
#include <climits>
#include <cmath>
#include <set>
#include <map>
#include <bitset>
using namespace std;
#define rep(i,n) for (int i = 0; i < (n); ++i)
using ll = long long;
using P = pair<int,int>;

template<class T>
using PQ = priority_queue<T,vector<T>,greater<T>>;

int main() {
  int n, m;
  cin >> n >> m;
  
  PQ<int> q;
  rep(i,n) q.push(i);
  PQ<P> events;

  vector<ll> ans(n);
  rep(mi,m) {
    int t, w, s;
    cin >> t >> w >> s;
    while (!events.empty() && events.top().first <= t) {
      q.push(events.top().second);
      events.pop();
    }
    if (q.empty()) continue;
    int i = q.top(); q.pop();
    ans[i] += w;
    events.emplace(t+s, i);
  }

  rep(i,n) cout << ans[i] << '\n';
  return 0;
}