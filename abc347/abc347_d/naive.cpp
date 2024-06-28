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
int main(){
  int a, b;
  long long C;
  cin >> a >> b >> C;
  int p = __builtin_popcountll(C);
  if (p > a + b || p < abs(a - b) || (a + b + p) % 2 != 0 || (a + b + p) / 2 > 60){
    cout << -1 << endl;
  } else {
    int x = (a + b - p) / 2;
    int y = a - x;
    int z = b - x;
    long long X = 0, Y = 0;
    for (int i = 0; i < 60; i++){
      if ((C >> i & 1) == 0){
        if (x > 0){
          X |= (long long) 1 << i;
          Y |= (long long) 1 << i;
          x--;
        }
      } else {
        if (z > 0){
          Y |= (long long) 1 << i;
          z--;
        } else {
          X |= (long long) 1 << i;
          y--;
        }
      }
    }
    cerr << x << " " << y << " " << z << endl;
    cout << X << ' ' << Y << endl;
  }
}