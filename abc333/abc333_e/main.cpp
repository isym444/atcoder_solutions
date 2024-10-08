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

using namespace std;

#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) ::std::begin(x), ::std::end(x)
#define ll long long
#define sz(x) (int)(x).size()
#define fo(from_0_to_non_incl_to) for(int i=0;i<from_0_to_non_incl_to;i++)
#define foi(from,non_incl_to) for(int i=from;i<non_incl_to;i++)
#define foj(from,non_incl_to) for(int j=from;j<non_incl_to;j++)
#define fok(from,non_incl_to) for(int k=from;k<non_incl_to;k++)
#define wasd(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define isvalid(x_plus_i,max_boundary_n,y_plus_j,max_boundary_m) (0<=x_plus_i and x_plus_i<max_boundary_n and 0<=y_plus_j and y_plus_j<max_boundary_m)
//#define gcd __gcd
#define mp make_pair
//Makes % get floor remainder (towards -INF) and make it always positive
#define MOD(x,y) (x%y+y)%y
#define print(p) cout<<p<<endl
#define fi first
#define sec second
#define prmap(m) {for(auto i: m) cout<<(i.fi)<<i.sec<<endl}
#define pra(a) {for(auto i: a) cout<<i<<endl;}
#define prm(a) {for(auto i: a) pra(i) cout<<endl;}
//#define itobin(x) bitset<32> bin(x)
#define itobin(intToConvertTo32BitBinaryNum) std::bitset<32>(intToConvertTo32BitBinaryNum)
#define bintoi(binaryNum32BitToConvertToInt) binaryNum32BitToConvertToInt.to_ulong()
#define binstoi(binaryStringToConvertToInt) stoi(binaryStringToConvertToInt, nullptr, 2)
#define vecsum(vectorName) accumulate((vectorName).begin(), (vectorName).end(), 0)
#define setbits(decimalnumber) __builtin_popcount(decimalnumber)

typedef pair<ll, ll> pl;

#define pb push_back

ll mod=1e9+7,INF=1e18;


/*/---------------------------IO(Debugging)----------------------/*/
template<class T> istream& operator >> (istream &is, vector<T>& V) {
    for(auto &e : V)
        is >> e;
    return is;
}

template<typename CharT, typename Traits, typename T>
ostream& _containerprint(std::basic_ostream<CharT, Traits> &out, T const &val) {
    return (out << val << " ");
}
template<typename CharT, typename Traits, typename T1, typename T2>
ostream& _containerprint(std::basic_ostream<CharT, Traits> &out, pair<T1, T2> const &val) {
    return (out << "(" << val.first << "," << val.second << ") ");
}
template<typename CharT, typename Traits, template<typename, typename...> class TT, typename... Args>
ostream& operator << (std::basic_ostream<CharT, Traits> &out, TT<Args...> const &cont) {
    out << "[ ";
    for(auto&& elem : cont) _containerprint(out, elem);
    return (out << "]");
}
template<class L, class R> ostream& operator << (ostream& out, pair<L, R> const &val){
    return (out << "(" << val.first << "," << val.second << ") ");
}
template<class P, class Q = vector<P>, class R = less<P> > ostream& operator << (ostream& out, priority_queue<P, Q, R> const& M){
    static priority_queue<P, Q, R> U;
    U = M;
    out << "{ ";
    while(!U.empty())
        out << U.top() << " ", U.pop();
    return (out << "}");
}
template<class P> ostream& operator << (ostream& out, queue<P> const& M){
    static queue<P> U;
    U = M;
    out << "{ ";
    while(!U.empty())
        out << U.front() << " ", U.pop();
    return (out << "}");
}

void setIO(string name = "")
{ // name is nonempty for USACO file I/O
    ios_base::sync_with_stdio(0);
    cin.tie(0); // see Fast Input & Output
    // alternatively, cin.tie(0)->sync_with_stdio(0);
    if (sz(name))
    {
    freopen((name + ".in").c_str(), "r", stdin); // see Input & Output
    freopen((name + ".out").c_str(), "w", stdout);
    }
}

const int MAXA = 5000006;
bool prime[MAXA];

void gen_primes() {
    fill(prime, prime+MAXA, true);
    prime[0] = 0;
    // prime[1] = 1 for our purposes
    for (int i = 4; i < MAXA; i += 2) {
        prime[i] = 0;
    }

    for (int p = 3; p*p < MAXA; p += 2) {
        if (!prime[p]) continue;
        for (int x = p*p; x < MAXA; x += p) {
            prime[x] = 0;
        }
    }
}

//visited nodes
vector<bool> v;
//bool vis[61][61][61][61]={0};
//graph as adjacency list
vector<vector<int> > g;

void edge(int a, int b)
{
    g[a].pb(b);
 
    // for undirected graph add this line
    // g[b].pb(a);
}

void bfs(int u)
{
    queue<int> q;
 
    q.push(u);
    v[u] = true;
 
    //If want first time something happens/node reached then break when that happens
    while (!q.empty()) {
 
        int f = q.front();
        q.pop();
 
        cout << f << " ";
 
        // Enqueue all adjacent of f and mark them visited 
        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!v[*i]) {
                q.push(*i);
                v[*i] = true;
            }
        }
    }
}

vector<int> parent; // To store parent information

vector<int> bfs(int start, int end) {
    queue<int> q;
    q.push(start);
    v[start] = true;
    parent[start] = -1; // Start node has no parent

    while (!q.empty()) {
        int f = q.front();
        q.pop();

        if (f == end) break; // Stop if we reach the end node

        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!v[*i]) {
                q.push(*i);
                v[*i] = true;
                parent[*i] = f; // Set parent
            }
        }
    }

    vector<int> path;
    for (int i = end; i != -1; i = parent[i]) {
        path.push_back(i);
    }
    reverse(path.begin(), path.end()); // Reverse to get the correct order
    return path;
}

string itobins(int n) {
    if (n == 0) return "0";

    string binary = "";
    while (n > 0) {
        binary += (n % 2) ? '1' : '0';
        n /= 2;
    }

    reverse(binary.begin(), binary.end()); // Reverse to get the correct order
    return binary;
}

string dtobx(int decimalNumber, int base) {
    if (base < 2 || base > 36) {
        return "Invalid base";
    }

    string result = "";
    while (decimalNumber > 0) {
        int remainder = decimalNumber % base;

        // Convert remainder to corresponding character
        if (remainder >= 10) {
            result += 'A' + (remainder - 10);
        } else {
            result += '0' + remainder;
        }

        decimalNumber /= base;
    }

    // Reverse the string as the result is calculated in reverse order
    reverse(result.begin(), result.end());

    return result.empty() ? "0" : result;
}

ll ceildiv(ll n, ll d){
    return((n+d-1)/d);
}

ll floordiv(ll n, ll d){
    ll x = (n%d+d)%d;
    return ((n-x)/d);
}

ll midpoint(ll L, ll R){
    return (L+(R-L)/2);
}

ll lcm(ll a, ll b) {
    return std::abs(a * b) / std::gcd(a, b);
}


int stringToBinary(const std::string& s, char charAsOne) {
    int x = 0;
    for (int j = 0; j < s.length(); j++) {
        x = 2 * x + (s[j] == charAsOne);
    }
    return x;
}

//returns index of first element greater than or equal to target
ll findGreaterEqual(vector<ll> sortedVector, ll target){
    auto it = lower_bound(sortedVector.begin(), sortedVector.end(), target);
    return it-sortedVector.begin();
}

//returns index of first element less than or equal to target
//if all elements are greater than target returns -1
//if all elements are smaller than target, returns last element
ll findLessEqual(vector<ll> sortedVector, ll target){
    auto it = upper_bound(sortedVector.begin(), sortedVector.end(), target);
    if(it != sortedVector.begin()){
        --it;
        if(*it<=target){
            return it-sortedVector.begin()+1;
        }
    }
    else{
        return -1;
    }
}

struct loc
{
    inline static ll x=0;
    inline static ll y=0;
    inline static char dir='r';
    //loc::x to access or modify x
};

//Graph visualizer:
//https://csacademy.com/app/graph_editor/

#define pb push_back
#include <unordered_map>

int main() {
    std::ios::sync_with_stdio(false);
    setIO("");
    std::cin.tie(nullptr);
    
    ll N;
    cin >> N;
    vector<ll> event;
    foi(0,N){
        ll t,x;
        cin >> t >> x;
        if(t==1){
            event.pb(x);
        }
        else{
            event.pb(-x);
        }
    }
    ll cur = 0;
    ll ans = 0;
    vector<ll> av;
    unordered_map<ll,ll> pot;
    for(int i = N-1; i>=0; i--){
        ll t = event[i];
        //if encounters a monster
        if(t<0){
            //incr pot type t he needs
            pot[-t]++;
            cur++;
        }
        else{
            //he encounters a potion
            //if he needs one to defeat a monster later on then --
            if(pot[t]>0){
                pot[t]--;
                cur--;
                av.pb(1);
            }
            else{
                av.pb(0);
            }
        }
        ans=max(ans,cur);
    }
    // foi(0,pot.size()){
    //     // cout << (*next(pot.begin(),i)).first << endl;
    //     if((*next(pot.begin(),i)).second>0){
    //         cout << -1 << endl;
    //         return 0;
    //     }
    // }
    for(auto x:pot){
        if(x.second>0){
            cout << -1 << endl;
            return 0;
        }
    }
    // if(cur>0){
    //     cout << -1 << endl;
    //     return 0;
    // }
    // cerr << av << endl;
    reverse(av.begin(),av.end());
    cout << ans << endl;
    for(auto x:av){
        cout << x << " ";
    }

    /* genprimes(1e5); */

    //Uncomment for BFS
    /* int n, e;
    //get number of nodes and edges
    cin >> n >> e;
    
    //initialize your visited vector and your graph as adjacency list
    v.assign(n, false);
    g.assign(n, vector<int>());
    
    parent.assign(n, -1);


    //construct your graph as an adjacency list
    int a, b;
    for (int i = 0; i < e; i++) {
        cin >> a >> b;
        edge(a, b);
    }
    
    //run the bfs and output order of traversed nodes
    for (int i = 0; i < n; i++) {
        if (!v[i])
            bfs(i);
    }
    cout << endl;
    
    //run the bfs outputing  path traversed + shortest path
    v.assign(n, false);
    int startNode, endNode;
    startNode = 0;
    endNode = 7;
    //cin >> startNode >> endNode;
    vector<int> path = bfs(startNode, endNode);

    // Output the path
    for (int node : path) {
        cout << node << " ";
    }
    cout << endl;
    // Output the length of the path i.e. # of edges
    cout << path.size()-1;
    cout << endl; */

    //Use for problems where you have to go up,down,left,right. Do x+i & y+j and i&j will test all 4 directions. Do x+i+1 & y+j+1 if 0 indexed
    /* wasd(
        //cout << "Use this for problems where you have to go up, down, left right" << endl;
    ) */
    return 0;
}
