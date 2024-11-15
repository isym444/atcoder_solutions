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
#include <cassert>
#include <unordered_map>
#include <type_traits> // For std::is_floating_point
#include <cmath> // For std::ceil
#include <iomanip>
#include <unordered_set>
#include <functional>
#include <type_traits>

using namespace std;

#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) ::std::begin(x), ::std::end(x)
#define ll long long
#define sz(x) (int)(x).size()
#define fo(from_0_to_non_incl_to) for(int i=0;i<from_0_to_non_incl_to;i++)
//h CAREFUL if you put an expression as an argument it will give bugs, better assign expression to variable then put that in the foi() as argument
#define foi(from,non_incl_to) for(int i=from;i<(non_incl_to);i++)
#define foii(non_incl_to) for(int i=0;i<(non_incl_to);i++)
#define foj(from,non_incl_to) for(int j=from;j<(non_incl_to);j++)
#define fojj(non_incl_to) for(int j=0;j<(non_incl_to);j++)
#define fok(from,non_incl_to) for(int k=from;k<(non_incl_to);k++)
#define fokk(non_incl_to) for(int k=0;k<(non_incl_to);k++)
#define fa(x, dataStructure) for(auto x : dataStructure)
#define fx(dataStructure) for(auto x : dataStructure)
#define wasd(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define qweasdzxc(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define isvalid(x_plus_i,max_boundary_n,y_plus_j,max_boundary_m) (0<=x_plus_i and x_plus_i<max_boundary_n and 0<=y_plus_j and y_plus_j<max_boundary_m)
//#define gcd __gcd
#define mp make_pair
//Makes % get floor remainder (towards -INF) and make it always positive
#define MOD(x,y) (x%y+y)%y
// #define print(p) cout<<p<<endl
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
#define stringSplice(str, i, j) (str).erase(i, j) //j is the length of string to erase starting from index i
#define string_pop_back(str) (str).pop_back()
#define substring(str, i, j) (str).substr(i, j) //j is the length of substring from i
#define rng(a) a.begin(),a.end()
#define all(a) a.begin(),a.end()

typedef pair<ll, ll> pl;
typedef vector<long long> vll;
typedef std::vector<std::vector<long long>> vvll;

#define pb push_back

//max heap priority queue i.e. top() gives largest value
typedef priority_queue<ll> maxpq;
//min heap priority queue i.e. top() gives smallest value
typedef priority_queue<ll, vector<ll>, greater<ll>> minpq;

//multiset provides automatic ordering on insertion but unlike set, keeps duplicate/multiple items of same value
//n.b. set also provides autoamtic ordering on insertion
//.count(x) O(num_of_x+logN)
//.find(x) O(logN) -> so use find over count if possible
//.insert(x) O(logN) -> inserts s.t. sorted order is maintained
//.erase(x) O(logN)
//begin() O(logN)
typedef multiset<ll> msll;
//doing mymultiset.erase(x) will erase all
#define mserasesingle(mymultiset, x) mymultiset.erase(mymultiset.find(x))
#define mseraseall(mymultiset, x) mymultiset.erase(x)
//find smallest and biggest elements O(1)
#define msmin(mymultiset) *mymultiset.begin()
#define msmax(mymultiset) *mymultiset.rbegin()



int main(){
    ll N;
    cin >> N;
    vll A(N);
    // cin >> A;
    foi(0,N){
        ll t;
        cin >> t;
        A[i]=t;
    }
    map<ll,ll> seensums;
    ll ans = 0;
    foi(0,N){
        ll sum = A[i]+i;
        ll dif = i-A[i];
        ans+=seensums[dif];
        seensums[sum]++;
    }
    // dbg(seensums);
    cout << ans << endl;
    return 0;
}