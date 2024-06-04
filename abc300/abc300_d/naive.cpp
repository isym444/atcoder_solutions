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
#include <deque>
#include <numeric>
#include <assert.h>
#include <unordered_map>
#include <type_traits> // For std::is_floating_point
#include <cmath> // For std::ceil


#define rep(i,n) for (int i = 0; i < (n); ++i)
using ll = long long;
using namespace std;

// Sieve of Eratosthenes
// https://youtu.be/UTVg7wzMWQc?t=2774
struct Sieve {
    int n;
    vector<int> f, primes;
    Sieve(int n=1):n(n), f(n+1) {
        f[0] = f[1] = -1;
        for (ll i = 2; i <= n; ++i) {
        if (f[i]) continue;
        primes.push_back(i);
        f[i] = i;
        for (ll j = i*i; j <= n; j += i) {
            if (!f[j]) f[j] = i;
        }
        }
    }
};

int main() {
    ll n;
    cin >> n;
    Sieve p(1e6);
    // for(int i = 0; i<10; i++){
    //     cerr << p.f[i] << endl;
    // }
    // cerr << endl;
    for(int i = 0; i<10; i++){
        // cerr << p.primes[i] << endl;
    }
    int ans = 0;
    int m = p.primes.size();
    cerr << m << endl;
    //for every prime a in p.primes
    rep(ai,m) {
        ll a = p.primes[ai];
        if (a*a*a*a*a > n) break;
        //for bi starting from ai+1
        for (int bi = ai+1; bi < m; bi++) {
            ll b = p.primes[bi];
            if (a*a*b*b*b > n) break;
            for (int ci = bi+1; ci < m; ci++) {
                ll c = p.primes[ci];
                if (a*a*b*c*c > n) break;
                ans++;
            }
        }
    }
    cout << ans << endl;
    return 0;
}