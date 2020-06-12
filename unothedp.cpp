

// unothedp.cpp : Defines the entry point for the console application.
//
#include <vector>
#include <stack>
#include <queue>
#include <iostream>
#include <math.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <numeric>

using namespace std;

#define BP (cout<<endl)

#define DOCK() do{                       \
                                  int dock;     \
                                  cin>>dock;    \
}while(0)

template<typename T>
void printContainer(vector<T> &container)
{
	for (auto it = container.begin(); it != container.end(); it++)
	{
		cout << *it;
		if (container.end() - 1 != it)
			cout << ", ";
	}
}

class Solution
{
public:
	// 121. Best Time to Buy and Sell Stock
	int maxProfit(vector<int>& prices)
	{
		if (1 >= prices.size())
			return 0;

		int mp = 0;
		int minCost = prices[0];

		// the max profit can only comes out with its minimal historical prices. 
		for (int i = 1; i < prices.size(); i++)
		{
			if (prices[i] < minCost) // means the price of the stock is falling, so we don't need to check the profit.
				minCost = prices[i];
			else if (prices[i] - minCost > mp)
				mp = prices[i] - minCost;
		}

		return mp;
	}

	// 122. Best Time to Buy and Sell Stock II
	int maxProfitII(vector<int>& prices)
	{
		if (1 >= prices.size())
			return 0;

		int mp = 0;

		for (int i = 1; i < prices.size(); i++)
		{
			if (0 < prices[i] - prices[i - 1])
				mp += prices[i] - prices[i - 1];
		}

		return mp;
	}

	// 309. Best Time to Buy and Sell Stock with Cooldown
	// int maxProfitWithCooldown(vector<int>& prices)
	// {
	//  int len = prices.size();
	//  int mp = 0;
	//  vector<int> dp1(len+1, 0), dp2(len+1, 0), dp3(len+1, 0), dp4(len+1, 0);

	//  for(int i = 1; i <= len; i++)
	//  {
	//      for(int j = i + 1; j <= len; j++)
	//      {
	//          int profit = prices[j-1] - prices[i-1];
	//          if(i >= 4)
	//              profit += dp1[i-2];
	//          dp4[j] = max(dp4[j-1], max(profit, dp3[j]));
	//          if(dp4[j] > mp)
	//              mp = dp4[j];
	//      }
	//      swap(dp1, dp2);
	//      swap(dp2, dp3);
	//      swap(dp3, dp4);
	//  }

	//  return mp;
	// }

	int maxProfitWithCooldown(vector<int>& prices)
	{
		if (1 >= prices.size())
			return 0;

		int len = prices.size();
		vector<int> buy(len, 0);
		vector<int> sell(len, 0);

		buy[0] = -prices[0];
		buy[1] = max(-prices[1], buy[0]);
		sell[1] = max(buy[0] + prices[1], sell[0]);

		for (int i = 2; i < len; i++)
		{
			buy[i] = max(sell[i - 2] - prices[i], buy[i - 1]);
			sell[i] = max(buy[i - 1] + prices[i], sell[i - 1]);
		}

		return sell.back();
	}

	// 714. Best Time to Buy and Sell Stock with Transaction Fee
	int maxProfitWithTransactionFee(vector<int>& prices, int fee)
	{
		if (1 >= prices.size())
			return 0;

		int len = prices.size();
		int buy = -prices[0] - fee;
		int sell = 0;

		for (int i = 1; i < len; i++)
		{
			int ystBuy = buy;
			buy = max(sell - prices[i] - fee, buy);
			sell = max(ystBuy + prices[i], sell);
		}

		return sell;
	}

	// 123. Best Time to Buy and Sell Stock III (at Most Two Transactions)
	int maxProfitAtMostTwoTransactions(vector<int>& prices)
	{
		if (1 >= prices.size())
			return 0;

		int n = prices.size();
		vector<int> dp1(5, INT_MIN);
		vector<int> dp2(5);

		dp1[0] = 0;

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				if (j % 2) //phase 1,3
				{
					dp2[j] = dp1[j - 1];
					if (i > 0 && dp1[j] != INT_MIN)
						dp2[j] = max(dp2[j], dp1[j] + prices[i] - prices[i - 1]);
				}
				else // phase 0,2,4
				{
					dp2[j] = dp1[j];
					if (i > 0 && j > 0 && dp1[j - 1] != INT_MIN)
						dp2[j] = max(dp2[j], dp1[j - 1] + prices[i] - prices[i - 1]);
				}
			}
			swap(dp1, dp2);
		}

		int mp = 0;
		for (int i = 0; i < 5; i += 2)
			mp = max(mp, dp1[i]);
		return mp;
	}

	// 188. Best Time to Buy and Sell Stock IV (at Most K Transactions)
	int maxProfitAtMostKTransactions(int k, vector<int>& prices)
	{
		int n = prices.size();
		if (n < 2)
			return 0;

		int mp = 0;

		if (k >= n / 2)
		{
			for (int i = 1; i < n; i++)
				if (prices[i] - prices[i - 1] > 0)
					mp += prices[i] - prices[i - 1];

			return mp;
		}

		int p = k * 2 + 1; // number of phases

		vector<int> dp1(p, INT_MIN);
		vector<int> dp2(p);

		dp1[0] = 0;

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < p; j++)
			{
				if (j % 2) // holding stock phase 1,3,5,...
				{
					dp2[j] = dp1[j - 1];
					if (i > 0 && dp1[j] != INT_MIN)
						dp2[j] = max(dp2[j], dp1[j] + prices[i] - prices[i - 1]);
				}
				else // no stock phase 0,2,4,...
				{
					dp2[j] = dp1[j];
					if (i > 0 && j > 0 && dp1[j - 1] != INT_MIN)
						dp2[j] = max(dp2[j], dp1[j - 1] + prices[i] - prices[i - 1]);
				}
			}

			swap(dp1, dp2);
		}

		for (int i = 0; i < p; i += 2)
			mp = max(mp, dp1[i]);

		return mp;
	}

	// 70. Climbing Stairs
	int climbStairs(int n)
	{
		if (0 >= n)
			return 0;

		vector<int> ways = { 1,2 };

		if (n > ways.size())
			for (int i = ways.size(); i < n; i++)
				ways.push_back(ways[i - 2] + ways[i - 1]);

		return ways[n - 1];
	}

	// 746. Min Cost Climbing Stairs
	int minCostClimbingStairs(vector<int>& cost)
	{
		if (cost.empty())
			return 0;
		if (1 == cost.size())
			return cost[0];

		vector<int> minCost;
		int len = cost.size() * 2;
		minCost.resize(len);
		minCost[0] = cost[0];
		minCost[1] = INT_MAX;
		minCost[2] = cost[0] + cost[1];
		minCost[3] = cost[1];
		len = 4;

		for (int i = 2; i < cost.size(); i++)
		{
			minCost[len] = cost[i] + min(minCost[len - 4], minCost[len - 2]);
			minCost[len + 1] = cost[i] + min(minCost[len - 3], minCost[len - 1]);
			len += 2;
		}

		return min(min(minCost.back(), minCost[len - 2]), min(minCost[len - 3], minCost[len - 4]));
	}

	// 413. Arithmetic Slices
	int numberOfArithmeticSlices(vector<int>& A)
	{
		int n = A.size();
		if (n < 3)
			return 0;

		int add = 0;
		vector<int> dp(n, 0);

		for (int i = 2; i < n; i++)
		{
			if (A[i] - A[i - 1] == A[i - 1] - A[i - 2])
			{
				dp[i] = dp[i - 1] + 1 + add;
				add++;
			}
			else
			{
				dp[i] = dp[i - 1];
				add = 0;
			}
		}

		return dp[n - 1];
	}

	// 300. Longest Increasing Subsequence
	// int lengthOfLIS(vector<int>& nums)
	// {
	// 	int n = nums.size();

	// 	if (0 == n)
	// 		return 0;

	// 	vector<int> dp(n, 1);
	// 	int ret = 1;

	// 	for (int i = 1; i < n; i++)
	// 	{
	// 		for (int j = 0; j < i; j++)
	// 		{
	// 			if (nums[i] > nums[j])
	// 			{
	// 				dp[i] = max(dp[i], dp[j] + 1);
	// 				ret = max(ret, dp[i]);
	// 			}
	// 		}
	// 	}

	// 	return ret;
	// }

	int binsrchIndex(int num, vector<int>& dp, int len)
	{
		int l = 0;
		int r = len - 1;

		while(l <= r)
		{
			int mid = l+(r-l)/2;

			if(num == dp[mid])
				return mid;
			else if(num < dp[mid])
				r = mid - 1;
			else
				l = mid + 1;
		}

		return l; // l always stays at the bigger one.
	}

	int lengthOfLIS(vector<int>& nums)
	{
		vector<int> dp;

		int len = 0;

		for(auto num : nums)
		{
			int index = binsrchIndex(num, dp, len);
			if(index == len)
			{
				dp.push_back(num);
				len++;
			}
			else
				dp[index] = num;
		}

		return len;
	}

	// 646. Maximum Length of Pair Chain
	static bool cmpPairs(const vector<int>& a, const vector<int>& b)
	{
		return a[0] < b[0];
	}

	int findLongestChain(vector<vector<int>>& pairs)
	{
		int n = pairs.size();
		
		if(0 == n)
			return 0;

		vector<int> dp(n, 1);
		int ret = 1;

		sort(pairs.begin(), pairs.end(), cmpPairs);

		for(int i = 1; i < n; i++)
		{
			for(int j = 0; j < i; j++)
			{
				if(pairs[j][1] < pairs[i][0])
				{
					dp[i] = max(dp[i], dp[j]+1);
					ret = max(ret, dp[i]);
				}
			}	
		}

        return ret;
    }

	// 376. Wiggle Subsequence
	// int wiggleMaxLength(vector<int>& nums)
	// {
	// 	int n = nums.size();

	// 	if(0 == n)
	// 		return 0;

	// 	vector<int> lw(n, 1);
	// 	vector<int> rw(n, 1);
	// 	int ret = 1;

	// 	for(int i = 1; i < n; i++)
	// 	{
	// 		for(int j = 0; j < i; j++)
	// 		{
	// 			if(nums[i] > nums[j])
	// 				rw[i] = max(rw[i], lw[j]+1);
	// 			else if(nums[i] < nums[j])
	// 				lw[i] = max(lw[i], rw[j]+1);
	// 			ret = max(ret, max(lw[i], rw[i]));
	// 		}
	// 	}

	// 	return ret;
    // }

	int wiggleMaxLength(vector<int>& nums)
	{
		int n = nums.size();

		if(0 == n)
			return 0;

		int lw = 1;
		int rw = 1;

		for(int i = 1; i < n; i++)
		{
			if(nums[i] > nums[i-1])
				rw = lw+1;
			else if(nums[i] < nums[i-1])
				lw = rw+1;
		}

		return max(lw, rw);
	}

	// 198. House Robber
	int rob(vector<int>& nums)
	{
		if (nums.empty())
			return 0;
		if (1 == nums.size())
			return nums[0];

		for (int i = 2; i < nums.size(); i++)
		{
			if (2 == i)
				nums[i] += nums[0];
			else if (1 != i)
				nums[i] += max(nums[i - 2], nums[i - 3]);
		}

		return max(nums.back(), nums[nums.size() - 2]);
	}

	// 213. House Robber II
	int robII(vector<int>& nums)
	{
		if (0 == nums.size())
			return 0;
		if (1 == nums.size())
			return nums.front();
		if (2 == nums.size())
			return max(nums.front(), nums.back());

		int len = nums.size();
		vector<int> maxRob(len * 2, 0);

		for (int i = 0; i < len; i++)
		{
			if (0 == i)
				maxRob[i] = nums[i];
			else if (1 == i)
				maxRob[i] = maxRob[len + i] = nums[i];
			else if (2 == i)
			{
				maxRob[i] = nums[i] + maxRob[i - 2];
				maxRob[len + i] = nums[i];
			}
			else if (len - 1 == i)
				maxRob[len + i] = nums[i] + max(maxRob[len + i - 2], maxRob[len + i - 3]);
			else
			{
				maxRob[i] = nums[i] + max(maxRob[i - 2], maxRob[i - 3]);
				maxRob[len + i] = nums[i] + max(maxRob[len + i - 2], maxRob[len + i - 3]);
			}
		}

		return max(max(maxRob[len - 2], maxRob[len - 3]), max(maxRob[len * 2 - 1], maxRob[len * 2 - 2]));
	}

	// 53. Maximum Subarray
	int maxSubArray(vector<int>& nums)
	{
		if (nums.empty())
			return 0;

		int sum = 0;
		int maxSum = INT_MIN;

		for (int i = 0; i < nums.size(); i++)
		{
			sum = max(sum + nums[i], nums[i]);
			maxSum = max(sum, maxSum);
		}

		return maxSum;
	}

	// 1218. Longest Arithmetic Subsequence of Given Difference
	int longestSubsequence(vector<int>& arr, int difference)
	{
		int maxSeqLen = 1;
		int foresee;

		unordered_map<int, int> dp;

		for (auto digit : arr)
		{
			foresee = digit + difference;
			if (dp[digit])
			{
				dp[foresee] = dp[digit] + 1;
				if (dp[foresee] > maxSeqLen)
					maxSeqLen = dp[foresee];
			}
			else
				dp[foresee] = 1;
		}

		return maxSeqLen;
	}

	// 392. Is Subsequence
	/*bool isSubsequence(string s, string t)
	{
	int startPos = 0;

	for (char c : s)
	{
	startPos = t.find(c, startPos);
	if (startPos == string::npos)
	return false;
	startPos++;
	}

	return true;
	}*/

	bool isSubsequence(string s, string t)
	{
		if (s.empty())
			return true;

		int k = 0;

		for (int i = 0; i < t.length(); i++)
		{
			if (s[k] == t[i])
				k++;
			if (s.length() == k)
				return true;
		}

		return false;
	}

	// 1143. Longest Common Subsequence
	/*int longestCommonSubsequence(string text1, string text2)
	{
	vector<vector<int>> dp(text1.length()+1, vector<int>(text2.length()+1, 0));

	for (int i = 0; i < text1.length(); i++)
	{
	for (int j = 0; j < text2.length(); j++)
	{
	if (text1[i] == text2[j])
	dp[i + 1][j + 1] = dp[i][j] + 1;
	else
	dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
	}
	}

	return dp[text1.length()][text2.length()];
	}*/

	int longestCommonSubsequence(string text1, string text2)
	{
		vector<vector<int>> dp(2, vector<int>(text2.length() + 1, 0));

		for (int i = 0; i < text1.length(); i++)
		{
			for (int j = 0; j < text2.length(); j++)
			{
				if (text1[i] == text2[j])
					dp[1][j + 1] = dp[0][j] + 1;
				else
					dp[1][j + 1] = max(dp[1][j], dp[0][j + 1]);
			}

			swap(dp[0], dp[1]);
		}

		return dp[0][text2.length()];
	}

	// 1092. Shortest Common Supersequence
	// T=O(n^2+2n^2)
	// string getLongestCommonSubsequence(string &s1, string &s2)
	// {
	//  vector<vector<string>> dp(s1.length()+1, vector<string>(s2.length()+1, ""));

	//  for(int i = 0; i < s1.length(); i++)
	//  {
	//      for(int j = 0; j < s2.length(); j++)
	//      {
	//          if(s1[i] == s2[j])
	//              dp[i+1][j+1] = dp[i][j] + s1[i];
	//          else
	//              dp[i+1][j+1] = (dp[i+1][j].length() > dp[i][j+1].length()) ? dp[i+1][j] : dp[i][j+1];
	//      }
	//  }

	//  return dp[s1.length()][s2.length()];
	// }

	// string shortestCommonSupersequence(string str1, string str2)
	// {
	//  if(str1.empty())
	//      return str2;
	//  else if(str2.empty())
	//      return str1;

	//  string lcs = getLongestCommonSubsequence(str1, str2);
	//  if(lcs.empty())
	//      return str1 + str2;

	//  string scs = "";
	//  int idx1 = 0;
	//  int idx2 = 0;

	//  for(auto c : lcs)
	//  {
	//      while(idx1 < str1.length())
	//      {
	//          if(str1[idx1] != c)
	//              scs.push_back(str1[idx1++]);
	//          else
	//          {
	//              idx1++;
	//              break;
	//          }
	//      }
	//      while(idx2 < str2.length())
	//      {
	//          if(str2[idx2] != c)
	//              scs.push_back(str2[idx2++]);
	//          else
	//          {
	//              idx2++;
	//              break;
	//          }
	//      }

	//      scs.push_back(c);
	//  }

	//  // scs.append(str1, idx1);
	//  // scs.append(str2, idx2);

	//  return scs + str1.substr(idx1) + str2.substr(idx2);
	// }

	// O(n^2+n)
	string shortestCommonSupersequence(string str1, string str2)
	{
		if (str1.empty())
			return str2;
		else if (str2.empty())
			return str1;

		int idx1 = str1.length();
		int idx2 = str2.length();

		vector<vector<int>> dp(str1.length() + 1, vector<int>(str2.length() + 1, 0));

		for (int i = 0; i < str1.length(); i++)
		{
			for (int j = 0; j < str2.length(); j++)
			{
				if (str1[i] == str2[j])
					dp[i + 1][j + 1] = dp[i][j] + 1;
				else
					dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
			}
		}

		if (0 == dp[idx1][idx2])
			return str1 + str2;

		deque<char> scs;
		char c;

		while (idx1 || idx2)
		{
			if (!idx1)
				c = str2[--idx2];
			else if (!idx2)
				c = str1[--idx1];
			else if (str1[idx1 - 1] == str2[idx2 - 1])
				c = str1[--idx1] = str2[--idx2];
			else if (dp[idx1][idx2] == dp[idx1 - 1][idx2])
				c = str1[--idx1];
			else if (dp[idx1][idx2] == dp[idx1][idx2 - 1])
				c = str2[--idx2];

			scs.push_front(c);
		}

		return{ begin(scs), end(scs) };
	}

	// 1062. Longest Repeating Substring todo(lock)


	// 5. Longest Palindromic Substring
	int getPalindromeLen(string s, int l, int r)
	{
		while (0 <= l && r < s.length() && s[l] == s[r])
			l--, r++;

		return r - l - 1;
	}

	string longestPalindrome(string s)
	{
		if (s.empty())
			return s;

		int start = 0, maxLen = 0;

		for (int i = 0; i < s.length(); i++)
		{
			// Expanding in both left and right direction, left increasement equals to right increasement.
			// If the number of remaining elements on the right is less than or equal to the half of maxLen, so it definitely can't increase to exceed maxLen.
			if (s.length() - i <= maxLen / 2)
				break;

			int len = max(getPalindromeLen(s, i, i), getPalindromeLen(s, i, i + 1));
			if (len > maxLen)
			{
				maxLen = len;
				start = i - (len - 1) / 2;
			}
		}

		return s.substr(start, maxLen);
	}

	// 516. Longest Palindromic Subsequence
	int longestPalindromeSubseq(string s)
	{
		if (s.empty())
			return 0;

		int len = s.length();
		vector<vector<int>> dp(2, vector<int>(len + 1, 0));

		for (int i = 0; i < len; i++)
		{
			for (int j = len - 1; j >= 0; j--)
			{
				if (s[i] == s[j])
					dp[1][len - j] = dp[0][len - j - 1] + 1;
				else
					dp[1][len - j] = max(dp[1][len - j - 1], dp[0][len - j]);
			}

			swap(dp[0], dp[1]);
		}

		return dp[0][len];
	}

	// 583. Delete Operation for Two Strings
	int minDeleteDistance(string word1, string word2)
	{
		int n1 = word1.size();
		int n2 = word2.size();

		if (0 == n1 && 0 == n2)
			return 0;
		else if (0 == n1)
			return n2;
		else if (0 == n2)
			return n1;

		vector<int> dp1(n2 + 1, 0);
		vector<int> dp2(n2 + 1, 0);

		for (int i = 0; i < n1; i++)
		{
			for (int j = 1; j <= n2; j++)
			{
				if (word1[i] == word2[j - 1])
					dp2[j] = dp1[j - 1] + 1;
				else
					dp2[j] = max(dp1[j], dp2[j - 1]);
			}
			swap(dp1, dp2);
		}

		return n1 + n2 - 2 * dp1[n2];
	}

	// 72. Edit Distance
	int minEditDistance(string word1, string word2)
	{
		int n1 = word1.size();
		int n2 = word2.size();

		if (0 == n1 && 0 == n2)
			return 0;
		else if (0 == n1)
			return n2;
		else if (0 == n2)
			return n1;

		vector<vector<int>> dp(n1, vector<int>(n2, 0));

		for (int i = 0; i < n1; i++)
		{
			if (word1[i] == word2[0])
				dp[i][0] = i;
			else
			{
				if (0 == i)
					dp[i][0] = 1;
				else
					dp[i][0] = dp[i - 1][0] + 1;
			}
		}

		for (int i = 0; i < n2; i++)
		{
			if (word2[i] == word1[0])
				dp[0][i] = i;
			else
			{
				if (0 == i)
					dp[0][i] = 1;
				else
					dp[0][i] = dp[0][i - 1] + 1;
			}
		}

		for (int i = 1; i < n1; i++)
		{
			for (int j = 1; j < n2; j++)
			{
				if (word1[i] == word2[j])
					dp[i][j] = dp[i - 1][j - 1];
				else
					dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
			}
		}

		return dp[n1 - 1][n2 - 1];
	}

	// 650. 2 Keys Keyboard
	// int minStepsWith2KeysKeyboard(int n)
	// {
	//  if(n < 2)
	//      return 0;

	//  vector<vector<int>> dp(n+1, vector<int>(n+1, INT_MAX));

	//  dp[1][0] = 0;
	//  dp[1][1] = 1;

	//  for(int i = 1; i < n; i++)
	//  {
	//      for(int j = 0; j <= i; j++)
	//      {
	//          if(dp[i][j] != INT_MAX)
	//          {
	//              if(i+j <= n)
	//                  dp[i+j][j] = min(dp[i+1][j], dp[i][j]+1); // dp[i+1][j] why is i+1???????
	//              dp[i][i] = min(dp[i][j]+1, dp[i][i]);
	//          }
	//      }
	//  }

	//  return *min_element(dp[n].begin(), dp[n].end());
	// }

	int minStepsWith2KeysKeyboard(int n)
	{
		if (n < 2)
			return 0;

		vector<int> dp1(n + 1, 0);
		vector<int> dp2(n + 1, 0);

		for (int i = 1; i <= n; i++)
			dp1[i] = i;

		for (int i = 2; i <= n / 2; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				dp2[j] = dp1[j];
				if (j != i && 0 == j%i)
					dp2[j] = min(dp2[j], dp2[i] + j / i);
			}
			swap(dp1, dp2);
		}

		return dp1[n];
	}

	// 416. Partition Equal Subset Sum
	// Backtracking
	bool canPartitionBtk(vector<int>& nums, int idx, int target)
	{
		if (0 == target)
			return true;

		if (idx == nums.size() || target < 0)
			return false;

		if (canPartitionBtk(nums, idx + 1, target - nums[idx]))
			return true;

		if (0 == idx)
			return false;

		int next = idx;
		while (++next < nums.size() && nums[next] == nums[idx]);

		return canPartitionBtk(nums, next, target);
	}

	// DP
	bool canPartitionDP(vector<int>& nums, int half)
	{
		vector<bool> dp(half + 1, false);

		dp[0] = true;

		for (int num : nums)
		{
			if (num > half)
				return false;

			for (int i = half; i >= num; i--)
			{
				if (dp[i - num])
				{
					if (i == half)
						return true;
					dp[i] = true;
				}
			}
		}

		return false;
	}

	bool canPartition(vector<int>& nums)
	{
		int n = nums.size();
		if (n < 2)
			return false;

		int sum = accumulate(nums.begin(), nums.end(), 0);

		if (sum % 2)
			return false;

		return canPartitionDP(nums, sum / 2);
		// return canPartitionBtk(nums, 0, sum / 2);
	}

	// 494. Target Sum
	// int findTargetSumWays(vector<int>& nums, int S)
	// {
	//  int n = nums.size();
	//  if (0 == n)
	//      return 0;

	//  if (1 == n)
	//      return (nums[0] == S || -nums[0] == S);

	//  int sum = accumulate(nums.begin(), nums.end(), 0);
	//  if (sum < S || (sum + S) % 2)
	//      return 0;

	//  int target = (S + sum) / 2;

	//  vector<int> dp(target + 1, 0);
	//  dp[0] = 1;

	//  for (int num : nums)
	//  {
	//      for (int i = target; i >= num; i--)
	//      {
	//          if (dp[i - num])
	//              dp[i] += dp[i - num];
	//      }
	//  }

	//  return dp[target];
	// }

	int findTargetSumWays(vector<int>& nums, int S)
	{
		int n = nums.size();

		if (0 == n)
			return 0;
		if (1 == n)
			return (nums[0] == S || -nums[0] == S);

		int sum = accumulate(nums.begin(), nums.end(), 0);
		if (S > sum)
			return 0;

		int originIdx = sum;
		sum *= 2;

		vector<int> dp1(sum + 1, 0);
		vector<int> dp2(sum + 1, 0);
		dp1[originIdx] = 1;

		for (int num : nums)
		{
			dp2.assign(sum + 1, 0);

			for (int i = 0; i <= sum; i++)
			{
				if (i - num >= 0 && i + num <= sum)
					dp2[i] = dp1[i - num] + dp1[i + num];
				else if (i - num >= 0)
					dp2[i] = dp1[i - num];
				else if (i + num <= sum)
					dp2[i] = dp1[i + num];
			}

			swap(dp1, dp2);
		}

		// for (int num : nums)
		// {
		//  dp2.assign(sum + 1, 0);

		//  for (int i = sum - num; i >= num; i--)
		//  {
		//      if (dp1[i])
		//      {
		//          dp2[i - num] += dp1[i];
		//          dp2[i + num] += dp1[i];
		//      }
		//  }

		//  swap(dp1, dp2);
		// }

		return dp1[originIdx + S];
	}

	// 474. Ones and Zeroes
	// int findMaxForm(vector<string>& strs, int m, int n)
	// {
	//     if(strs.empty())
	//      return 0;

	//  vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

	//  for(auto str : strs)
	//  {
	//      int zeros = count(str.begin(), str.end(), '0');
	//      int ones = count(str.begin(), str.end(), '1');

	//      for(int i = m; i >= zeros ; i--)
	//      {
	//          for(int j = n; j >= ones ; j--)
	//          {
	//              dp[i][j] = max(dp[i][j], dp[i-zeros][j-ones]+1);
	//          }
	//      }
	//  }

	//  return dp[m][n];
	// }

	int findMaxForm(vector<string>& strs, int m, int n)
	{
		if (strs.empty())
			return 0;

		vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MIN));

		dp[0][0] = 0;
		int ret = 0;

		for (auto str : strs)
		{
			int zeros = count(str.begin(), str.end(), '0');
			int ones = count(str.begin(), str.end(), '1');

			for (int i = m; i >= zeros; i--)
			{
				for (int j = n; j >= ones; j--)
				{
					if (dp[i - zeros][j - ones] != INT_MIN)
					{
						dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1);
						ret = max(ret, dp[i][j]);
					}
				}
			}
		}

		return ret;
	}

	// AcWing 2. 01 Bag Question
	int maxWorthFor01Bag(vector<int>& v, vector<int>& w, int V)
	{
		int n = v.size();

		vector<int> dp(V+1, 0);

		for(int i = 0; i < n; i++)
		{
			for(int j = V; j >= v[i]; j--)
			{
				dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
			}
		}

		return dp[V];
	}

	// 322. Coin Change
	// int coinChange(vector<int>& coins, int amount)
	// {
	// 	int n = coins.size();

	// 	vector<vector<int>> dp(n + 1, vector<int>(amount + 1, INT_MAX));
	// 	dp[0][0] = 0;

	// 	for (int i = 1; i <= n; i++)
	// 	{
	// 		int coin = coins[i - 1];

	// 		for (int j = 0; j <= amount; j++)
	// 		{
	// 			dp[i][j] = dp[i - 1][j];
	// 			if(j < coin)
	// 				continue;

	// 			for (int k = 1; k <= j/coin; k++)
	// 			{
	// 				int x = j - (coin*k);
	// 				if (dp[i - 1][x] != INT_MAX)
	// 					dp[i][j] = min(dp[i][j], dp[i - 1][x] + k);
	// 			}
	// 		}
	// 	}

	// 	return (INT_MAX == dp[n][amount])?-1:dp[n][amount];
	// }

	int coinChange(vector<int>& coins, int amount)
	{
		vector<int> dp(amount + 1, INT_MAX);
		dp[0] = 0;

		for (auto coin : coins)
		{
			for (int i = coin; i <= amount; i++)
			{
				if (dp[i - coin] != INT_MAX)
					dp[i] = min(dp[i], dp[i - coin] + 1);
			}
		}

		return (INT_MAX == dp[amount]) ? -1 : dp[amount];
	}

	// 518. Coin Change 2
	// int change(int amount, vector<int>& coins)
	// {
	// 	vector<int> dp1(amount+1, 0);
	// 	vector<int> dp2(amount+1, 0);
	// 	dp1[0] = 1;

	// 	for(auto coin : coins)
	// 	{
	// 		for(int i = 0; i <= amount; i++)
	// 		{
	// 			if(i < coin)
	// 				dp2[i] = dp1[i];
	// 			else
	// 				dp2[i] = dp1[i]+ dp2[i-coin];
	// 		}

	// 		swap(dp1, dp2);
	// 	}

	// 	return dp1[amount];
	// }

	int change(int amount, vector<int>& coins)
	{
		vector<int> dp(amount + 1, 0);
		dp[0] = 1;

		for (auto coin : coins)
		{
			for (int i = coin; i <= amount; i++)
			{
				dp[i] += dp[i - coin];
			}
		}

		return dp[amount];
	}

	// AcWing 3. Complete Bag Quesiton
	int maxWorthForCompleteBag(vector<int>& v, vector<int>& w, int V)
	{
		int n = v.size();

		vector<int> dp(V+1, 0);

		for(int i = 0; i < n; i++)
		{
			for(int j = v[i]; j <= V; j++)
			{
				dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
			}
		}

		return dp[V];
	}

	// 139. Word Break
	bool wordBreak(string s, vector<string>& wordDict)
	{
		int n = s.length();
		vector<bool> dp(n + 1, false);
		dp[0] = true;

		for (int i = 1; i <= n; i++)
		{
			for (auto word : wordDict)
			{
				int len = word.length();
				if (i >= len && s.substr(i - len, len) == word)
				{
					dp[i] = dp[i] || dp[i - len];
				}
			}
		}

		return dp[n];
	}

	// 377. Combination Sum IV
	int combinationSum4(vector<int>& nums, int target)
	{
		vector<unsigned long> dp(target + 1, 0);
		dp[0] = 1;

		sort(nums.begin(), nums.end());

		for (int i = 1; i <= target; i++)
		{
			for (auto num : nums)
			{
				if (i >= num)
					dp[i] += dp[i - num];
			}
		}

		return dp[target];
	}

	// AcWing 4. Multiple Bag Quesiton
	/*
		0<N,V≤100
		0<vi,wi,si≤100

		N: number of types of goods
		V: volume of this bag
		si: number of goods of each type

		T=O(N*V*si)
		T=100*100*100=10^6
	*/
	int maxWorthForMultipleBag(vector<int>& v, vector<int>& w, vector<int>& s, int V)
	{
		int n = v.size();

		vector<int> dp(V+1, 0);

		for(int i = 0; i < n; i++)
		{
			for(int j = V; j >= v[i]; j--)
			{
				for(int k = 1; k <= s[i] && k <= j/v[i]; k++)
				{
					dp[j] = max(dp[j], dp[j-k*v[i]] + k*w[i]);
				}
			}
		}

		return dp[V];
	}

	// AcWing 5. Multiple Bag Question II
	/*
		0<N≤1000
		0<V≤2000
		0<vi,wi,si≤2000

		N: number of types of goods
		V: volume of this bag
		si: number of goods of each type

		if we take the same algorithm as previous quesion T=O(N*V*si), T=1000*2000*2000=10^9, time exceeds
		so we should optimize the time complexity of algorithm to T=O(N*V*lg2(si)), T=1000*2000*lg2(2000)=10^7
	*/
	int maxWorthForMultipleBagII(vector<int>& v, vector<int>& w, vector<int>& s, int V)
	{
		int n = v.size();

		vector<int> bv;  
		vector<int> bw;

		for(int i = 0; i < n; i++)
		{
			int num = s[i];
			for(int j = 1; j <= num; j*=2)
			{
				num -= j;
				bv.push_back(v[i]*j);
				bw.push_back(w[i]*j);
			}
			if(num > 0)
			{
				bv.push_back(v[i]*num);
				bw.push_back(w[i]*num);
			}
		}

		return maxWorthFor01Bag(bv, bw, V);
	}

	// AcWing 6. Multiple Bag Question III
	/*
		0<N≤1000
		0<V≤20000
		0<vi,wi,si≤20000

		N: number of types of goods
		V: volume of this bag
		si: number of goods of each type

		If we take the same algorithm as previous quesion T=O(N*V*lg2(si)), T=1000*20000*lg2(20000)>10^8, time exceeds
		so we should optimize the time complexity of algorithm to T=O()
	*/
	int maxWorthForMultipleBagIII(vector<int>& v, vector<int>& w, vector<int>& s, int V)
	{
		int n = v.size();

		vector<int> f(V+1, 0);
		vector<int> idx(V+1, 0);
		vector<int> q(V+1, 0);

		for(int i = 0; i < n; i++)
		{
			/*
				In previous questions, j is always increasing from 0 to V, the number of remainders of j/v is v, the remainders are in the range from 0 to v-1,
				for example v = 4, r = 0,1,2,3;
				classify j into v groups according to the number of remainders of j/v, 
				so that the solution is:
					for example: j = 18, v = 4 -> f[18] = max(f[14]+w, f[10]+2w, f[6]+3w, f[2]+4w)
					f[j] = max(f[j-1*v] + 1*w, f[j-2*v] + 2*w, f[j-3*v] + 3*w, ..., f[j-k*v] + k*w)
	  				f[j+v] = max(f[j] + 1*w, f[j-1*v] + 2*w, f[j-2*v] + 3*w, ..., f[j-k*v] + (k+1)*w)
				this is a monotonous queue question. A question about the monotonous queue is referring to the leetcode 239. Sliding Window Maximum
				you know, f[2] will add 4w, f[6] will only add 3w, so we can think of this series as a series of {f[0], f[v]-w, f[2v]-2w, f[3v]-3w, ..., f[kv]-kw}
				f[j] is the max of the series of {f[0], f[v]-w, f[2v]-2w, f[3v]-3w, ..., f[kv]-kw}
			*/
			for(int j = 0; j < v[i]; j++)
			{
				int le = 0, ri = -1;
				for(int k = j; k <= V; k+=v[i])
				{
					// template for monotonous queue question
					while(le <= ri && q[ri] < f[k] - k/v[i]*w[i])
						ri--;
					ri++;
					q[ri] = f[k] - k/v[i]*w[i];
					idx[ri] = k;
					if(idx[le] + s[i]*v[i] < k)
						le++;
					f[k] = q[le] + idx[ri]/v[i]*w[i];
				}
			}
		}

		return f[V];
	}

	// AcWing 7. Compound Bag Question
	int maxWorthForCompoundBag(vector<int>& v, vector<int>& w, vector<int>& s, int V)
	{
		int n = v.size();

		vector<int> cv;
		vector<int> cw;
		vector<int> cs;

		for(int i = 0; i < n; i++)
		{
			if(s[i] < 0) // 0/1 bag
			{
				cv.push_back(v[i]);
				cw.push_back(w[i]);
				cs.push_back(-1);
			}
			else if(0 == s[i]) // complete bag
			{
				cv.push_back(v[i]);
				cw.push_back(w[i]);
				cs.push_back(0);
			}
			else // multiple bag
			{
				for(int x = 1; x < s[i]; x *= 2)
				{
					s[i] -= x;
					cv.push_back(x * v[i]);
					cw.push_back(x * w[i]);
					cs.push_back(-1);
				}
				if(s[i] > 0)
				{
					cv.push_back(s[i] * v[i]);
					cw.push_back(s[i] * w[i]);
					cs.push_back(-1);
				}
			}
		}

		n = cv.size();
		vector<int> dp(V+1, 0);

		for(int i = 0; i < n; i++)
		{
			if(-1 == cs[i]) // 0/1 bag
			{
				for(int j = V; j >= cv[i]; j--)
					dp[j] = max(dp[j], dp[j - cv[i]] + cw[i]);
			}
			else // complete bag
			{
				for(int j = cv[i]; j <= V; j++)
					dp[j] = max(dp[j], dp[j - cv[i]] + cw[i]);
			}
		}

		return dp[V];
	}

	// AcWing 8. Two Dimensional Bag Question
	int maxWorthForTwoDimensionalBag(vector<int>& v, vector<int>& m, vector<int>& w, int V, int M)
	{
		int n = v.size();

		vector<vector<int>> dp(V+1, vector<int>(M+1, 0));
		
		for(int i = 0; i < n; i++)
		{
			for(int j = V; j >= v[i]; j--)
			{
				for(int k = M; k >= m[i]; k--)
				{
					dp[j][k] = max(dp[j][k], dp[j-v[i]][k-m[i]]+w[i]);
				}
			}
		}

		return dp[V][M];
	}

	// AcWing 9. Grouping Bag Question
	int maxWorthForGroupingBag(vector<vector<int>>& v, vector<vector<int>>& w, int V)
	{
		int N = v.size();

		vector<int> dp(V+1, 0);

		for(int i = 0; i < N; i++)
		{
			int n = v[i].size();
			
			for(int k = V; k >= 0; k--)
			{
				for(int j = 0; j < n; j++)
				{
					if(k >= v[i][j])
					{
						dp[k] = max(dp[k], dp[k-v[i][j]]+w[i][j]);
					}
				}
			}
		}

		return dp[V];
	}

	// AcWing 10. Dependence Bag Question
	int maxWorthForDependenceBag()
	{
		return 0;
	}

	// AcWing 11. Number of Solutions of Bag Question
	int maxNumberOfSolutionsOfBag(vector<int>& v, vector<int>& w, int V)
	{
		int mod = 1000000009;
		int n = v.size();

		vector<int> f(V+1, INT_MIN);
		vector<int> g(V+1, 0);

		f[0] = 0;
		g[0] = 1;

		for(int i = 0; i < n; i++)
		{
			for(int j = V; j >= v[i]; j--)
			{
				int t = max(f[j], f[j-v[i]]+w[i]);
				int s = 0;
				if(t == f[j])
					s += g[j];
				if(t == f[j-v[i]]+w[i])
					s += g[j-v[i]];
				if(s >= mod)
					s -= mod;
				
				f[j] = t;
				g[j] = s;
			}
		}

		int maxw = 0;
		for(int i = 0; i <= V; i++)
			maxw = max(maxw, f[i]);

		int res = 0;
		for(int i = 0; i <= V; i++)
		{
			if(maxw == f[i])
			{
				res += g[i];
				if(res >= mod)
					res -= mod;
			}
		}

		return res;
	}

	// AcWing 12. Solution of Bag Question
	void solutionOfBag(vector<int>& v, vector<int>& w, int V)
	{
		int n = v.size();

		vector<vector<int>> dp(n+1, vector<int>(V+1, 0));

		for(int i = n-1; i >=0; i--)
		{
			for(int j = 1; j <= V; j++)
			{
				dp[i][j] = dp[i+1][j];
				if(j >= v[i])
					dp[i][j] = max(dp[i][j], dp[i+1][j-v[i]]+w[i]);
			}
		}

		for(int i = 0; i < n; i++)
		{
			if(dp[i][V] == dp[i+1][V-v[i]]+w[i])
			{
				cout << i+1 << " ";
				V -= v[i];
			}
		}
	}

	// 62. Unique Paths
	// int uniquePaths(int m, int n)
	// {
	//     if(m <= 0 || n <= 0)
	//      return 0;

	//  vector<vector<long>> dp(m+1, vector<long>(n+1, 0));
	//  dp[0][1] = 1;

	//  for(int i = 1; i <= m; i++)
	//  {
	//      for(int j = 1; j <= n; j++)
	//      {
	//          dp[i][j] = dp[i-1][j] + dp[i][j-1];
	//      }
	//  }

	//  return dp[m][n];
	// }

	int uniquePaths(int m, int n)
	{
		if (m <= 0 || n <= 0)
			return 0;

		vector<long> dp1(n + 1, 0);
		vector<long> dp2(n + 1, 0);

		dp1[1] = 1;

		for (int i = 0; i < m; i++)
		{
			for (int j = 1; j <= n; j++)
				dp2[j] = dp1[j] + dp2[j - 1];

			swap(dp1, dp2);
		}

		return dp1[n];
	}

	// 64. Minimum Path Sum
	int minPathSum(vector<vector<int>>& grid)
	{
		int m = grid.size();
		if (0 == m)
			return 0;

		int n = grid[0].size();
		if (0 == n)
			return 0;

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (0 == i && 0 == j)
					continue;
				else if (0 == i)
					grid[i][j] += grid[i][j - 1];
				else if (0 == j)
					grid[i][j] += grid[i - 1][j];
				else
					grid[i][j] += min(grid[i - 1][j], grid[i][j - 1]);
			}
		}

		return grid[m - 1][n - 1];
	}

	// 63. Unique Paths II
	// int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
	// {
	//  int m = obstacleGrid.size();
	//  if(0 == m)
	//      return 0;

	//  int n = obstacleGrid[0].size();
	//  if(0 == n)
	//      return 0;

	//  if(1 == obstacleGrid[0][0] || 1 == obstacleGrid[m-1][n-1])
	//      return 0;

	//  vector<vector<long>> dp(m, vector<long>(n, 0));

	//  for(int i = 0; i < m; i++)
	//  {
	//      for(int j = 0; j < n; j++)
	//      {
	//          if(1 == obstacleGrid[i][j])
	//              continue;
	//          if(0 == i && 0 == j)
	//              dp[i][j] = 1;
	//          else if(0 == i)
	//              dp[i][j] = dp[i][j-1];
	//          else if(0 == j)
	//              dp[i][j] = dp[i-1][j];
	//          else
	//              dp[i][j] = dp[i-1][j] + dp[i][j-1];
	//      }
	//  }

	//  return dp[m-1][n-1];
	// }

	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
	{
		int m = obstacleGrid.size();
		if (0 == m)
			return 0;

		int n = obstacleGrid[0].size();
		if (0 == n)
			return 0;

		if (1 == obstacleGrid[0][0] || 1 == obstacleGrid[m - 1][n - 1])
			return 0;

		vector<long> dp1(n + 1, 0);
		vector<long> dp2(n + 1, 0);

		dp1[1] = 1;

		for (int i = 0; i < m; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				if (1 == obstacleGrid[i][j - 1])
					dp2[j] = 0;
				else
					dp2[j] = dp1[j] + dp2[j - 1];
			}
			swap(dp1, dp2);
		}

		return dp1[n];
	}

	// 887. Super Egg Drop
	//int binsrchFirstEggDropFloor(vector<vector<int>>& dp, int k, int n)
	//{
	//  int l = 0;
	//  int r = n - 1;
	//  int minimal = dp[k - 1][0];

	//  while (l <= r)
	//  {
	//      int m = l + (r - l) / 2;

	//      if (dp[k][m] < dp[k - 1][n - 1 - m])
	//          l = m + 1;
	//      else if (dp[k][m] > dp[k - 1][n - 1 - m])
	//      {
	//          minimal = dp[k][m];
	//          r = m - 1;
	//      }
	//      else
	//          return dp[k][m];
	//  }

	//  return minimal;
	//}

	//int superEggDrop(int K, int N)
	//{
	//  if (1 == K || N < 3)
	//      return N;

	//  vector<vector<int>> dp(K + 1, vector<int>(N + 1, INT_MAX));

	//  for (int i = 0; i <= N; i++)
	//  {
	//      dp[0][i] = 0;
	//      dp[1][i] = i;
	//  }

	//  for (int i = 0; i <= K; i++)
	//      dp[i][0] = 0;

	//  for (int k = 2; k <= K; k++)
	//  {
	//      for (int n = 1; n <= N; n++)
	//      {
	//          /*for (int x = 1; x <= n; x++)
	//              dp[k][n] = min(dp[k][n], max(dp[k - 1][x - 1], dp[k][n - x]) + 1);*/

	//          dp[k][n] = binsrchFirstEggDropFloor(dp, k, n) + 1;
	//      }
	//  }

	//  return dp[K][N];
	//}

	// What the maximum number of floors can be tested with K eggs and M moves
	int superEggDrop(int K, int N)
	{
		if (1 == K || N < 3)
			return N;

		vector<vector<int>> dp(K + 1, vector<int>(N + 1, INT_MIN));

		for (int i = 0; i <= N; i++)
		{
			dp[0][i] = 0;
			dp[1][i] = i;
		}

		for (int i = 0; i <= K; i++)
			dp[i][0] = 0;

		for (int k = 2; k <= K; k++)
		{
			for (int m = k; m <= N; m++)
			{
				if (m == k)
					dp[k][m] = dp[k - 1][m - 1] * 2 + 1;
				else
					dp[k][m] = dp[k - 1][m - 1] + 1 + dp[k][m - 1];

				if (dp[k][m] >= N)
				{
					if (k == K || m == k)
						return m;
					break;
				}
			}
		}

		return 0;
	}
};

// 303. Range Sum Query - Immutable
class NumArray {
public:
	vector<int> m_sums;

	NumArray(vector<int>& nums)
	{
		if (nums.empty())
			return;

		m_sums.push_back(nums[0]);

		for (int i = 1; i < nums.size(); i++)
			m_sums.push_back(nums[i] + m_sums[i - 1]);
	}

	int sumRange(int i, int j)
	{
		return (0 == i) ? m_sums[j] : m_sums[j] - m_sums[i - 1];
	}
};

// 304. Range Sum Query 2D - Immutable
// class NumMatrix {
// public:
//     vector<vector<int>> m_rowSums;

//  NumMatrix(vector<vector<int>>& matrix)
//  {
//      for(int row = 0; row < matrix.size(); row++)
//      {
//          m_rowSums.emplace_back();
//          m_rowSums[row].push_back(matrix[row][0]);

//          for(int col = 1; col < matrix[row].size(); col++)
//              m_rowSums[row].push_back(matrix[row][col] + m_rowSums[row][col-1]);
//      }
//  }

//  int sumRegion(int row1, int col1, int row2, int col2)
//  {
//      int sum = 0;

//      for(int row = row1; row <= row2; row++)
//          sum += (0 == col1)?m_rowSums[row][col2]:(m_rowSums[row][col2] - m_rowSums[row][col1-1]);

//      return sum;
//  }
// };

// sum of right-top corner. T=O(n^3)+O(1)
// class NumMatrix {
// public:
//  vector<vector<int>> m_rectSums;

//  NumMatrix(vector<vector<int>>& matrix)
//  {
//      if(matrix.empty() || matrix.front().empty())
//          return;

//      m_rectSums.push_back(vector<int>(matrix.front().size()+1,0));

//      for(int row = 0; row < matrix.size(); row++)
//      {
//          m_rectSums.push_back({0});

//          for(int col = 0; col < matrix[row].size(); col++)
//          {
//              m_rectSums[row+1].push_back(matrix[row][col] + m_rectSums[row+1].back());
//              for(int downRow = row+1; downRow < matrix.size(); downRow++)
//                  m_rectSums[row+1][col+1] += matrix[downRow][col];
//          }
//      }

//      m_rectSums.push_back(vector<int>(matrix.front().size()+1,0));
//     }

//     int sumRegion(int row1, int col1, int row2, int col2)
//  {
//      return m_rectSums[row1+1][col2+1] - m_rectSums[row1+1][col1] - m_rectSums[row2+2][col2+1] + m_rectSums[row2+2][col1];
//     }
// };

// sum of right-bottom corner. T=O(n^2)+O(1)
class NumMatrix {
public:
	vector<vector<int>> m_rbCornerSums;

	NumMatrix(vector<vector<int>>& matrix)
	{
		if (matrix.empty() || matrix[0].empty())
			return;

		m_rbCornerSums.push_back(vector<int>(matrix[0].size() + 1, 0));

		for (int row = 0; row < matrix.size(); row++)
		{
			m_rbCornerSums.push_back({ 0 });

			for (int col = 0; col < matrix[row].size(); col++)
				m_rbCornerSums[row + 1].push_back(matrix[row][col] + m_rbCornerSums[row + 1][col] + m_rbCornerSums[row][col + 1] - m_rbCornerSums[row][col]);
		}
	}

	int sumRegion(int row1, int col1, int row2, int col2)
	{
		return m_rbCornerSums[row2 + 1][col2 + 1] - m_rbCornerSums[row1][col2 + 1] - m_rbCornerSums[row2 + 1][col1] + m_rbCornerSums[row1][col1];
	}
};

int main()
{
	Solution solu;

	// 121. Best Time to Buy and Sell Stock
	// vector<int> prices = { 7,1,5,3,6,4 };
	// // prices = { 7,6,4,3,1 };
	// // prices = {};
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfit(prices) << endl << endl;

	// 122. Best Time to Buy and Sell Stock II
	// vector<int> prices = { 7,1,5,3,6,4 };
	// // prices = { 7,6,4,3,1 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfitII(prices) << endl << endl;

	// 309. Best Time to Buy and Sell Stock with Cooldown
	// vector<int> prices = { 7,1,5,3,6,4 };
	// // prices = { 1,2,3,0,2 };
	// // prices = { 2,1,2,1,0,0,1 };
	// prices = { 1,2,4,2,5,7,2,4,9,0 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfitWithCooldown(prices) << endl << endl;

	// 714. Best Time to Buy and Sell Stock with Transaction Fee
	// vector<int> prices = { 1, 3, 2, 8, 4, 9 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// int fee = -1;
	// while(0 > fee)
	// {
	//  cout << "Transaction fee: ";
	//  cin >> fee;
	// }
	// cout << "Max profit: " << solu.maxProfitWithTransactionFee(prices, fee) << endl << endl;

	// 123. Best Time to Buy and Sell Stock III (at Most Two Transactions)
	// vector<int> prices = { 3,3,5,0,0,3,1,4 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfitAtMostTwoTransactions(prices) << endl << endl;

	// 188. Best Time to Buy and Sell Stock IV (at Most K Transactions)
	/*vector<int> prices = { 3,3,5,0,0,3,1,4 };
	prices = { 2,4,1 };
	prices = { 3,2,6,5,0,3 };
	cout << "prices: [ ";
	printContainer(prices);
	cout << " ]" << endl;
	int k = -1;
	while (k <= 0)
	{
	cout << "Transacions: ";
	cin >> k;
	}
	cout << "Max profit: " << solu.maxProfitAtMostKTransactions(k, prices) << endl << endl;*/

	// 70. Climbing Stairs
	// int n;
	// while (1)
	// {
	//  cout << "How many stairs do you wanna climb: ";
	//  cin >> n;
	//  cout << "For total " << n << " stairs, distinct ways: " << solu.climbStairs(n) << endl << endl;
	// }

	// 746. Min Cost Climbing Stairs
	// vector<int> cost;
	// cost = { 10, 15, 20 };
	// //cost = { 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
	// cost = { 1, 0, 0, 0 };
	// cout << "Minimum  cost: " << solu.minCostClimbingStairs(cost) << endl << endl;

	// 413. Arithmetic Slices
	// vector<int> A = { 3, -1, -5, -9 };
	// cout << "A: [ ";
	// printContainer(A);
	// cout << " ]" << endl;
	// cout << "Number of arithmetic slices: " << solu.numberOfArithmeticSlices(A) << endl << endl;

	// 300. Longest Increasing Subsequence
	// vector<int> nums = { 10,9,2,5,3,7,101,18 };
	// cout << "Length of longest increasing subsequence: " << solu.lengthOfLIS(nums) << endl << endl;

	// 646. Maximum Length of Pair Chain
	// vector<vector<int>> pairs;
	// pairs = {
	// 	{ 1,2 },
	// 	{ 2,3 },
	// 	{ 3,4 }
	// };

	// pairs = {
	// 	{ 3,4 },
	// 	{ 2,3 },
	// 	{ 1,2 }
	// };

	// cout << "Length of the longest chain is: " << solu.findLongestChain(pairs) << endl << endl;

	// 376. Wiggle Subsequence
	// vector<int> nums = { 1,7,4,9,2,5 };
	// nums = { 1,17,5,10,13,15,10,5,16,8 };
	// nums = { 1,2,3,4,5,6,7,8,9 };
	// cout << "Length of the longest wiggle subsequence is: " << solu.wiggleMaxLength(nums) << endl << endl;

	// 198. House Robber
	// vector<int> nums;
	// nums = { 1,2,3,1 };
	// nums = { 2,7,9,3,1 };
	// nums = { 1,2,3,1,7,6,5,9,2,2,6 };
	// cout << "Money amount list: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Maximum amount of robbery: " << solu.rob(nums) << endl << endl;

	// 213. House Robber II
	// vector<int> nums;
	// nums = { 1,2,3,1 };
	// nums = { 2,7,9,3,1 };
	// //nums = { 1,2,3,1,7,6,5,9,2,2,6 };
	// cout << "Money amount list: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Maximum amount of robbery: " << solu.robII(nums) << endl << endl;

	// 53. Maximum Subarray
	// vector<int> nums = { -2,1,-3,4,-1,2,1,-5,4 };
	// cout << "Maximum Subarray: " << solu.maxSubArray(nums) << endl << endl;

	// 303. Range Sum Query - Immutable
	// vector<int> nums;
	// nums = { -2, 0, 3, -5, 2, -1 };
	// int i,j; // indices i and j (i <= j)
	// NumArray* na = new NumArray(nums);

	// while(1)
	// {
	//  cout << "Input index i: ";
	//  cin >> i;
	//  while(1)
	//  {
	//      cout << "Input index j: ";
	//      cin >> j;
	//      if(i <= j)
	//          break;
	//  }
	//  cout << "The sum of elements between indices " << i << " and " << j << ": " << na->sumRange(i, j) << endl << endl;
	// }

	// 304. Range Sum Query 2D - Immutable
	// vector<vector<int>> matrix;
	// matrix = {
	//  { 3, 0, 1, 4, 2 },
	//  { 5, 6, 3, 2, 1 },
	//  { 1, 2, 0, 1, 5 },
	//  { 4, 1, 0, 1, 7 },
	//  { 1, 0, 3, 0, 5 }
	// };
	// int row1, col1, row2, col2;
	// NumMatrix* nm = new NumMatrix(matrix);

	// while(1)
	// {
	//  cout << "row1: ";
	//  cin >> row1;
	//  cout << "col1: ";
	//  cin >> col1;
	//  while(1)
	//  {
	//      cout << "row2: ";
	//      cin >> row2;
	//      if(row1 <= row2)
	//          break;
	//  }
	//  while(1)
	//  {
	//      cout << "col2: ";
	//      cin >> col2;
	//      if(col1 <= col2)
	//          break;
	//  }
	//  cout << "Sum of region ((" << row1 << "," << col1 << "),(" << row2 << "," << col2 << ")): " << nm->sumRegion(row1, col1, row2, col2) << endl << endl;
	// }

	// 1218. Longest Arithmetic Subsequence of Given Difference
	/*vector<int> arr;
	arr = { 1,2,3,4 };
	arr = { 1,5,7,8,5,3,4,2,1 };
	int difference;

	while (1)
	{
	cout << "Difference: ";
	cin >> difference;
	cout << "The lengh of longest arithmetic subsequence for difference " << difference << " is: " << solu.longestSubsequence(arr, difference) << endl << endl;
	}*/


	// 392. Is Subsequence
	/*string t = "ahbgdc";
	string s;

	while (1)
	{
	cout << "String t: " << t << endl;
	cout << "Inuput string s: ";
	cin >> s;
	cout << "s is a subsequence of t: " << (solu.isSubsequence(s, t) ? "true" : "false") << endl << endl;
	}*/

	// 1143. Longest Common Subsequence
	/*string text1 = "abcde";
	string text2 = "ace";
	cout << "String1: " << text1 << endl;
	cout << "String2: " << text2 << endl;
	cout << "Length of longest common subsequence is: " << solu.longestCommonSubsequence(text1, text2) << endl << endl;*/

	// 1092. Shortest Common Supersequence
	// string str1 = "cijkchc";
	// string str2 = "hcijkc";
	// cout << "String1: " << str1 << endl;
	// cout << "String2: " << str2 << endl;
	// cout << "Shortest Common Supersequence: " << solu.shortestCommonSupersequence(str1, str2) << endl << endl;

	// 1062. Longest Repeating Substring todo(lock)


	// 5. Longest Palindromic Substring
	// string s = "babad";
	// s = "atsgstsgstkpobbvijklmnnmlkjiqr";
	// cout << "String: " << s << endl;
	// cout << "Longest Palindromic Substring: " << solu.longestPalindrome(s) << endl << endl;

	// 516. Longest Palindromic Subsequence
	// string s = "babadbqwer";
	// s = "abcdefgecba";
	// // s = "bbbab";
	// cout << "String: " << s << endl;
	// cout << "Longest Palindromic Subsequence: " << solu.longestPalindromeSubseq(s) << endl << endl;

	// 583. Delete Operation for Two Strings
	// string word1 = "sea";
	// string word2 = "eat";
	// word1 = "intention";
	// word2 = "execution";
	// cout << "word1: " << word1 << endl;
	// cout << "word2: " << word2 << endl;
	// cout << "Minamal distance: " << solu.minDeleteDistance(word1, word2) << endl << endl;

	// 72. Edit Distance
	// string word1 = "horse";
	// string word2 = "ros";
	// word1 = "intention";
	// word2 = "execution";
	// cout << "word1: " << word1 << endl;
	// cout << "word2: " << word2 << endl;
	// cout << "Minimal edit distance: " << solu.minEditDistance(word1, word2) << endl << endl;

	// 650. 2 Keys Keyboard
	// while(1)
	// {
	//  int n = 0;
	//  while(n <= 0)
	//  {
	//      cout << "Number of 'A': ";
	//      cin >> n;
	//  }
	//  cout << "Minimal steps with 2 keys keyboard: " << solu.minStepsWith2KeysKeyboard(n) << endl << endl;
	// }


	/*
		0/1 Bag
	*/

	// 416. Partition Equal Subset Sum
	// vector<int> nums = { 1, 5, 11, 5 };
	// nums = { 1,1,2,5,5,5,5 };
	// nums = { 1, 3, 5 };
	// cout << "nums: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Can partition: " << (solu.canPartition(nums) ? "true" : "false") << endl << endl;

	// 494. Target Sum
	// vector<int> nums = { 1, 5, 11, 5 };
	// // nums = { 1,1,2,5,5,5,5 };
	// // nums = { 1, 3, 5 };
	// nums = { 1, 1, 1, 1, 1 };
	// // nums = { 1, 0 };
	// cout << "nums: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// int S;
	// while (1)
	// {
	//  cout << "Target Sum: ";
	//  cin >> S;
	//  cout << "Number of ways to get target sum: " << solu.findTargetSumWays(nums, S) << endl << endl;
	// }

	// 474. Ones and Zeroes
	// vector<string> strs = { "10", "0001", "111001", "1", "0" };
	// while(1)
	// {
	//  int m = -1;
	//  int n = -1;

	//  while(m < 0)
	//  {
	//      cout << "m: ";
	//      cin >> m;
	//  }

	//  while(n < 0)
	//  {
	//      cout << "n: ";
	//      cin >> n;
	//  }

	//  cout << "Maximum number of strings: " << solu.findMaxForm(strs, m, n) << endl << endl;
	// }

	// AcWing 2. 01 Bag Question
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<int> v(N, 0);
	// vector<int> w(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// }

	// cout << "Max worth: " << solu.maxWorthFor01Bag(v, w, V) << endl << endl;

	/*
		Complete Bag
	*/

	// 322. Coin Change
	// vector<int> coins = { 1,2,5 }; // 11->3
	// coins = { 186,419,83,408 }; // 6249->20
	// while (1)
	// {
	// 	int amount = 0;

	// 	while (amount <= 0)
	// 	{
	// 		cout << "amount: ";
	// 		cin >> amount;
	// 	}

	// 	cout << "Fewest number of coins: " << solu.coinChange(coins, amount) << endl << endl;
	// }

	// 518. Coin Change 2
	// vector<int> coins = { 1,2,5 }; // 5->4
	// coins = { 2 }; // 3->0
	// coins = { 186,419,83,408 }; // 6249->19
	// while (1)
	// {
	// 	int amount = 0;

	// 	while (amount <= 0)
	// 	{
	// 		cout << "amount: ";
	// 		cin >> amount;
	// 	}

	// 	cout << "Number of combinations: " << solu.change(amount, coins) << endl << endl;
	// }

	// AcWing 3. Complete Bag Quesiton
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<int> v(N, 0);
	// vector<int> w(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// }

	// cout << "Max worth: " << solu.maxWorthForCompleteBag(v, w, V) << endl << endl;

	/*
		Ordered Complete Bag
	*/

	// 139. Word Break
	/*string s = "leetcode";
	vector<string> wordDict = { "leet", "code" };
	cout << "Can be segmented: " << (solu.wordBreak(s, wordDict) ? "true" : "false") << endl << endl;*/

	// 377. Combination Sum IV
	/*vector<int> nums = { 1, 2, 3 };
	while (1)
	{
		int target = 0;
		
		while (target <= 0)
		{
			cout << "target: ";
			cin >> target;
		}

		cout << "Number of combinations: " << solu.combinationSum4(nums, target) << endl << endl;
	}*/

	/*
		Multiple Bag
	*/

	// AcWing 4. Multiple Bag Quesiton
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<int> v(N, 0);
	// vector<int> w(N, 0);
	// vector<int> s(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// 	cout << "Number of the " << i+1 << " good: ";
	// 	cin >> s[i];
	// }

	// cout << "Max worth: " << solu.maxWorthForMultipleBag(v, w, s, V) << endl << endl;
	
	// AcWing 5. Multiple Bag Question II
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<int> v(N, 0);
	// vector<int> w(N, 0);
	// vector<int> s(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// 	cout << "Number of the " << i+1 << " good: ";
	// 	cin >> s[i];
	// }

	// cout << "Max worth: " << solu.maxWorthForMultipleBagII(v, w, s, V) << endl << endl;

	// AcWing 6. Multiple Bag Question III
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<int> v(N, 0);
	// vector<int> w(N, 0);
	// vector<int> s(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// 	cout << "Number of the " << i+1 << " good: ";
	// 	cin >> s[i];
	// }

	// cout << "Max worth: " << solu.maxWorthForMultipleBagIII(v, w, s, V) << endl << endl;

	// AcWing 7. Compound Bag Question
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<int> v(N, 0);
	// vector<int> w(N, 0);
	// vector<int> s(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// 	cout << "Number of the " << i+1 << " good: ";
	// 	cin >> s[i];
	// }

	// cout << "Max worth: " << solu.maxWorthForCompoundBag(v, w, s, V) << endl << endl;

	// AcWing 8. Two Dimensional Bag Question --- V: volume limitation of bag, M: weight limitation of bag
	// int N, V, M;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;
	// cout << "Weight of bag: ";
	// cin >> M;

	// vector<int> v(N, 0);
	// vector<int> m(N, 0);
	// vector<int> w(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Weight of the " << i+1 << " good: ";
	// 	cin >> m[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// }

	// cout << "Max worth: " << solu.maxWorthForTwoDimensionalBag(v, m, w, V, M) << endl << endl;

	// AcWing 9. Grouping Bag Question
	// int N, V;
	// cout << "Number of groups of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;

	// vector<vector<int>> v;
	// vector<vector<int>> w;

	// for(int i = 0; i < N; i++)
	// {
	// 	int n;
	// 	cout << "Number of goods of the " << i+1 << " group: ";
	// 	cin >> n;

	// 	v.push_back(vector<int>(n, 0));
	// 	w.push_back(vector<int>(n, 0));

	// 	for(int j = 0; j < n; j++)
	// 	{
	// 		cout << "Volume of the " << j+1 << " good: ";
	// 		cin >> v[i][j];
	// 		cout << "Worth of the " << j+1 << " good: ";
	// 		cin >> w[i][j];
	// 	}
	// }

	// cout << "Max worth: " << solu.maxWorthForGroupingBag(v, w, V) << endl << endl;

	// https://www.bilibili.com/video/av34467850/?p=2
	// AcWing 10. Dependence Bag Question --- Hard --- involve DFS

	// AcWing 11. Number of Solutions of Bag Question
	// int N, V;
	// cout << "Number of goods: ";
	// cin >> N;
	// cout << "Volume of bag: ";
	// cin >> V;
	
	// vector<int> v(N, 0);
	// vector<int> w(N, 0);

	// for(int i = 0; i < N; i++)
	// {
	// 	cout << "Volume of the " << i+1 << " good: ";
	// 	cin >> v[i];
	// 	cout << "Worth of the " << i+1 << " good: ";
	// 	cin >> w[i];
	// }

	// cout << "Max number of solutions: " << solu.maxNumberOfSolutionsOfBag(v, w, V) << endl << endl;

	// AcWing 12. Solution of Bag Question
	int N, V;
	cout << "Number of goods: ";
	cin >> N;
	cout << "Volume of bag: ";
	cin >> V;
	
	vector<int> v(N, 0);
	vector<int> w(N, 0);

	for(int i = 0; i < N; i++)
	{
		cout << "Volume of the " << i+1 << " good: ";
		cin >> v[i];
		cout << "Worth of the " << i+1 << " good: ";
		cin >> w[i];
	}

	solu.solutionOfBag(v, w, V);

	// 62. Unique Paths
	// while(1)
	// {
	//  int m = 0;
	//  int n = 0;

	//  while (m <= 0)
	//  {
	//      cout << "m: ";
	//      cin >> m;
	//  }

	//  while (n <= 0)
	//  {
	//      cout << "n: ";
	//      cin >> n;
	//  }

	//  cout << "Number of unique paths: " << solu.uniquePaths(m, n) << endl << endl;
	// }

	// 64. Minimum Path Sum
	// vector<vector<int>> grid;
	// grid = {
	//  { 1,3,1 },
	//  { 1,5,1 },
	//  { 4,2,1 }
	// };
	// cout << "Grid:" << endl;
	// for(auto row : grid)
	// {
	//  cout << "[ ";
	//  printContainer(row);
	//  cout << " ]" << endl;
	// }
	// cout << "Minimum Path Sum: " << solu.minPathSum(grid) << endl << endl;

	// 63. Unique Paths II
	// vector<vector<int>> obstacleGrid;
	// obstacleGrid = {
	//  { 0,0,0 },
	//  { 0,1,0 },
	//  { 0,0,0 }
	// };

	// cout << "obstacleGrid:" << endl;
	// for(auto row : obstacleGrid)
	// {
	//  cout << "[ ";
	//  printContainer(row);
	//  cout << " ]" << endl;
	// }
	// cout << "Number of unique paths: " << solu.uniquePathsWithObstacles(obstacleGrid) << endl << endl;

	// 887. Super Egg Drop
	// while (1)
	// {
	//  int K = 0, N = 0;
	//  while (K <= 0)
	//  {
	//      cout << "Number of eggs: ";
	//      cin >> K;
	//  }
	//  while (N <= 0)
	//  {
	//      cout << "Number of floors: ";
	//      cin >> N;
	//  }

	//  cout << "Minimal number of moves: " << solu.superEggDrop(K, N) << endl << endl;
	// }

	DOCK();

	return 0;
}

