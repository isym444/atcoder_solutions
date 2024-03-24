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

using namespace std;

//for iterating over possible directions from a square in a 2d array -> for both wasd & including diagonals
vector<int> dx = {1, 0, -1, 0, 1, 1, -1, -1};
vector<int> dx_wasd = {1,-1,0,0};
vector<int> dy = {0, 1, 0, -1, 1, -1, 1, -1};
vector<int> dy_wasd = {0,0,1,-1};

//Graph visualizer:
//https://csacademy.com/app/graph_editor/

typedef long long int ll;

#define rep(i,n) for(int i = 0; i < (n); ++i)
#define drep(i,n) for(int i = (n)-1; i >= 0; --i)
// const int MX = 200005;
const int MX = 30;
const ll LINF = 1001002003004005006ll;
#define mins(x,y) (x = min(x,y))
#define chmin(x,y) x=min(x,y)
#define chmax(x,y) x=max(x,y)



// ll dp[MX];

// int main() {
//     int n,w;
//     scanf("%d%d",&n,&w);
//     cerr << "N: " << n << " W: " << w << endl;
//     rep(i,MX) dp[i] = 100000;
//     dp[0] = 0;
//     rep(i,n) {
//         int a,b;
//         int wi,vi;
//         scanf("%d%d",&wi,&vi);
//         drep(j,MX-vi) {
//             dp[j+vi] = min(dp[j+vi], dp[j]+wi);
//         }
//     }
//     int ans = 0;
//     // cerr << dp << endl;
//     int ii=0;
//     for(auto x:dp){
//         cerr << ii << ": " << x << endl;
//         ii++;
//     }
//     rep(i,MX) if (dp[i] <= w) ans = i;
//     cerr << "ans: " << ans << endl;
//     cout<<ans<<endl;
//     return 0;
// }

int main(){
    int N,W;
	cin>>N>>W;
	// int inf = 100;
	ll inf = 1e3;
    // ll dp[101][100001];
	// int V = 100000;
    int V = 20;
    ll dp[N+1][V+1];
    // vector<vector<int>> dp(N+1, vector<int>(V+1));
	rep(i,N+1) rep(j,V+1) dp[i][j] = inf;
	dp[0][0] = 0;
    for(int i = 0; i<N; i++){
        for(int j = 0; j<V; j++){
            cerr << dp[i][j] << " ";
        }
        cerr << endl;
    }
    cerr << endl;
	rep(i,N){
		int w,v;
		cin>>w>>v;
		rep(j,V+1) if(dp[i][j] != inf){
			chmin(dp[i+1][j],dp[i][j]);
			chmin(dp[i+1][j+v],dp[i][j]+w);
            for(int i = 0; i<N; i++){
                for(int j = 0; j<V; j++){
                    cerr << dp[i][j] << " ";
                }
            cerr << endl;
		}
        cerr << endl;
    }
	}
	int ans = 0;

    cerr << endl;
    for(int i = 0; i<N; i++){
        for(int j = 0; j<V; j++){
            cerr << dp[i][j] << " ";
        }
        cerr << endl;
    }
	rep(j,V+1) if(dp[N][j]<=W) chmax(ans,j);
	cout<<ans<<endl;
    return 0;
}
