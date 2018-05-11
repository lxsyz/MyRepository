import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import javax.xml.soap.Node;

public class LeetCode {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		ListNode node1 = new ListNode(1);
		ListNode pListNode = new ListNode(5);
		node1.next = pListNode;
		ListNode node2 = new ListNode(9);
		pListNode.next = node2;
		node2.next = new ListNode(2);
		node2.next.next = new ListNode(3);
//		System.out.println(node1.next.val);
//		node2.next = new ListNode(9);
		LeetCode leetCode = new LeetCode();
//		leetCode.addTwoNumbers(node1, node2);
//		leetCode.LongestSub("abcdab");
//		System.out.println(leetCode.letterCombination("23451"));
	}

	/**
	 * 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
		输出：7 -> 0 -> 8
		原因：342 + 465 = 807
	 * 测试用例： 1-》8 + 0   输出  1， 8
	 *         5 + 5   输出   0  1
	 *  从头到尾遍历两个链表，短的链表的后面的数用0填充，直到长的链表也走到了尽头，并且进位为0
	 * @author Administrator
	 *
	 */
	public ListNode addTwoNumbers(ListNode p1, ListNode p2) {
		ListNode res = new ListNode(0);
		ListNode temp = res;
		
		int plus = 0;
		int value = 0;
		while (p1 != null && p2 != null) {
			value = p1.val + p2.val + plus;
			
			if (value >= 10) {
				plus = 1;
				value = value % 10;
			} else
				plus = 0;
			
			temp.next = new ListNode(value);
			p1 = p1.next;
			p2 = p2.next;
			temp = temp.next;
			
			if (p1 == null && p2 == null)
				break;
			
			if (p1 == null)
				p1 = new ListNode(0);
			if (p2 == null)
				p2 = new ListNode(0);
			
		}
		
		if (plus == 1) {
			temp.next = new ListNode(1);
		}
		return res.next;
	}
	
	/**
	 * 简化版的代码，在讨论区看到的
	 * 思路是一样的，
	 * @param l1
	 * @param l2
	 * @return
	 */
	public ListNode simplifyAddTwoNumber(ListNode l1, ListNode l2) {
		ListNode res = new ListNode(0);
		ListNode temp = res;
		int extra = 0;
		while (l1 != null || l2 != null || extra != 0) {
			int sum = (l1 != null ? l1.val:0) + (l2 != null? l2.val:0) + extra;
			extra = sum / 10;
			temp.next = new ListNode(sum % 10);
			temp = temp.next;
			l1 = l1!=null ? l1.next:l1;
			l2 = l2!=null ? l2.next:l2;
		}
		return res.next;
	}

	/**
	 * 最长子串的长度
	 * 示例： abca    acbdbe
	 * 1. 将每个字符存入到哈希表中，索引存放其后一位的索引值
	 * 2. 当读取到重复的字符时，取出哈希表中的字符索引，即对应着重复字符的后一位
	 * 3. 计算当前长度，并与历史长度进行比较，取最大长度
	 * @param s
	 * @return
	 */
	public int lengthOfLongestSubstring(String s) {
		HashMap<Character, Integer> map = new HashMap<>();
		char[] arr = s.toCharArray();
		int maxlength = 0;
		int j = 0; 
		for (int i = 0;i < arr.length;i++) {
			if (map.containsKey(arr[i])) {
				j = Math.max(map.get(arr[i]), j);
			}
			maxlength = Math.max(maxlength, i - j + 1);
			map.put(arr[i], i + 1);
		}
		return maxlength;
	}
	
	public int lengthOfLongestSubstring2(String s) {
		char[] arr = s.toCharArray();
		int length = arr.length;
		int maxLength = 0;
		int lens = 1;
		int index = 1;
		for (int i = 0;i < length;i++) {
			// 假设该字符已经进入
			lens++;
			for (int j = index;j < i;j++) {
				// 判断前面是否有和它相同的字符
				if (arr[i] == arr[j]) {
					// 更新最大长度
					if (maxLength < lens - 1) {
						maxLength = lens- 1;
					}
					// 跳过中间字符
					index = j + 1;
					// 重置长度       如abcdb,重置为3
					lens = i - j;
					break;
				}	
			}
		}
		return Math.max(maxLength, lens);
	}
	
	/**
	 * 最长公共子串
	 * 动态规划
	 * @param s1
	 * @param s2
	 * @return
	 */
	public ArrayList<String> LCS(String s1, String s2) {
		int m = s1.length();
		int n = s2.length();
		int len = 0;
//		StringBuilder sb = new StringBuilder();
//		String res = "";
		ArrayList<String> list = new ArrayList<>();
		int[][] L = new int[n + 1][m+1];
		for (int i = 1;i < n+1;i++) {
			for (int j = 1;j < m + 1;j++) {
				if (s1.charAt(j) == s2.charAt(i)) {
					if (i == 1 || j == 1)
						L[i][j] = 1;
					else
						L[i][j] = L[i-1][j-1] + 1;
					if (L[i][j] > len) {
						len = L[i][j];
						list.clear();
						list.add(s1.substring(i - len + 1, i + 1));
					} else if (L[i][j] == len){
						list.add(s1.substring(i-len + 1, i + 1));
					}
				} else {
					L[i][j] = 0;
				}
			}
		}
		return list;
	}
	
	/**
	 * 最大回文子串
	 * 遍历字符串，计算字符左右字符是否相等
	 * @param s
	 * @return
	 */
	public String LongestPalindrome(String s) {
		int start = 0, end = 0;
		int len = s.length();
		for (int i = 0;i < len;i++) {
			int len1 = calcLen(s, i, i);
			// 考虑abba等情况
			int len2 = calcLen(s, i, i + 1);
			int res = Math.max(len1, len2);
			if (res > end - start) {
				start = i - (res - 1) / 2;
				end = start + res - 1;
			}
		}
		return s.substring(start, end + 1);
	}
	
	public int calcLen(String s, int left, int right) {
		while (left >= 0 && right <= s.length() - 1 && s.charAt(left) == s.charAt(right)) {
			left--;
			right++;
		}
		return right-left-1;
	}
	
	/**
	 * 寻找规律 Zigzag conversion
	 * 第一行：间隔为  nums * 2 - 2
	 * 之后每一行都有个 interval - 2 * j的数要插入
	 * @param s
	 * @param numRows
	 * @return
	 */
	public String convert(String s, int numRows) {
		if (numRows == 1) {
			return s;
		}
		int interval = numRows * 2 - 2;
		StringBuilder sb = new StringBuilder();
		int length = s.length();
		for (int j = 0;j < numRows;j++) {
			for (int i = j;i < length;i = i+interval) {
				if (j == numRows - 1 || j == 0)
					sb.append(s.charAt(i));
				else {
					sb.append(s.charAt(i));
					int leftInterval = interval - 2 * j;
					if (i + leftInterval < length)
						sb.append(s.charAt(i + leftInterval));
				}
			}
		}
		return sb.toString();
	}
	
	public int reverse(int x) {
		int result = 0;
		while (x != 0) {
			int tail = x % 10;
			int newResult = result * 10 + tail;
			// if overflows
			if ((newResult - tail) / 10 != result)
				return 0;
			result = newResult;
			x = x / 10;
		}
		return result;
	}
	
	public boolean isMatch(String s, String pattern) {
		if (s == null || pattern == null)
			return false;
		
		return matchCore(s, 0, pattern, 0);
	}
	/**
	 * 1. 模式先到尾部，匹配失败,两个都到达尾部，匹配成功
	 * 2. 第一种情况，字符相匹配，且模式后者为 '*'， 分三种情况递归：
	 * （1） 字符串加1，模式不变，表示模式匹配到了字符
	 * （2） 当字符串走到了末尾，模式还没有走到，回溯，模式做加2操作 
	 * （3）字符串不变，模式加2. 表示模式匹配到了0个字符
	 * 3. 第二种情况，字符不相匹配或者字符串读到了结尾，且模式后者为'*'
	 *   则模式加2，递归
	 * 4. 第三种情况，字符相匹配且模式后一字符不为'*'，
	 *   则模式加1，字符加1
	 * 5. 其余情况下返回false;
	 * 
	 * @return
	 */
	public boolean matchCore(String s, int sIndex, String pattern, int patternIndex) {
		if (sIndex == s.length() - 1 && patternIndex == pattern.length() - 1) 
			return true;
		if (sIndex != s.length() - 1 && patternIndex == pattern.length() - 1)
			return false;
		
		if (patternIndex + 1 < pattern.length() && pattern.charAt(patternIndex + 1) == '*') { 
			
			if (sIndex < s.length() && (pattern.charAt(patternIndex) == s.charAt(sIndex) || 
			    pattern.charAt(patternIndex) == '.')) {
                    return matchCore(s, sIndex + 1, pattern, patternIndex) ||
                            matchCore(s, sIndex, pattern, patternIndex + 2);
			} else {
                return matchCore(s, sIndex, pattern, patternIndex + 2);
            }
		} else {
			if (sIndex < s.length() && patternIndex < pattern.length() && (pattern.charAt(patternIndex) == s.charAt(sIndex) || 
			pattern.charAt(patternIndex) == '.')) {
				return matchCore(s, sIndex + 1, pattern, patternIndex + 1);
			} 
		}
		return false;
	}
	
	/**
	 * 水的最大面积
	 * 假设height[m]与height[n]间值最大，那么m的左边不存在更大的值，n的右边也不会存在更大的值
	 * 
	 * @param height
	 * @return
	 */
	public int maxArea(int[] height) {
		int length = height.length;
		int area = 0;
		int left = 0;
		int right = length - 1;
		int maxarea = 0;
		while (left != right) {
			if (height[left] < height[right]) {
				area = (right - left) * height[left];
				left++;
			} else {
				area = (right - left) * height[right];
				right--;
			}
			maxarea = Math.max(area, maxarea);
		}
		
		
		return maxarea;
	}
	
	/**
	 * Symbol       Value
		I             1
		V             5
		X             10
		L             50
		C             100
		D             500
		M             1000
		易理解的暴力法，可AC
	 * @param num
	 * @return
	 */
	public String intToRoman(int num) {
		HashMap<Integer, String> map = new HashMap<>();
		map.put(1, "I");
		map.put(4, "IV");
		map.put(5, "V");
		map.put(9, "IX");
		map.put(10, "X");
		map.put(40, "XL");
		map.put(90, "XC");
		map.put(50, "L");
		map.put(100, "C");
		map.put(400, "CD");
		map.put(500, "D");
		map.put(900, "CM");
		map.put(1000, "M");
		int digit = 1;
		int digit5 = 5;
		int digit4 = 4;
		int digit9 = 9;
		String result = "";
		
		while (num != 0) {
            int number = 0;
			String temp = "";
			int tail = num % 10;
			while (tail < 4 && number < tail) {
				temp = temp + map.get(digit); 
				number++;
			}
			if (tail > 4 && tail < 9) {
                temp = temp + map.get(digit5);
                while (number < tail - 5) {
				    temp = temp + map.get(digit);
				    number++;
                }
			}
			if (tail == 4) {
				temp = temp + map.get(digit4);
			} else if (tail == 9) {
				temp = temp + map.get(digit9); 
			}
			result = temp + result;
			num = num / 10;
			digit *= 10;
			digit5 *= 10;
			digit4 *= 10;
			digit9 *= 10;
		}
		
		return result;
	}
	
	/**
	 * 简单的方法
	 * @param num
	 * @return
	 */
	public String intToRoman2(int num) {
		String[] roman = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V","IV", "I"};
        int[] nums = {1000, 900, 500, 400,  100,  90,   50,  40,  10,  9,   5 , 4, 1};
        StringBuilder sb = new StringBuilder();
        // 从首位往个位做
        for (int i = 0;i < nums.length;i++) {
        	int count = num / nums[i];
        	while (count > 0) {
        		sb.append(roman[i]);
        		count--;
        	}
        	// 去掉已经做过的首位
        	num = num % nums[i];
        }
        String string = "sad";
        return sb.toString();
	}
	
	/**
	 * 常规解法
	 * 1. 先找到最短的那个字符串
	 * 2. 遍历，逐个比较字符,更新最短的字符串
	 * 好的解法：
	 * 1. 外层循环，遍历字符串字符  for i ; size()>0;i++
	 * 2. 内层循环，遍历所有字符串，后面的字符串与前面的字符串比较单个字符      strs[j][i] != strs[j-1][i]
	 *    如果字符符合条件，添加到prefix中， prefix += strs[0][i]
	 *    直到i > strs[j].size()大于其中某个字符串长度。
	 * @param strs
	 * @return
	 */
	public String longestCommonPrefix(String[] strs) {
		String shortest = strs[0];
		int length = strs[0].length();
		for (int i = 0;i < strs.length;i++) {
			if (strs[i].length() < length) {
				length = strs[i].length();
				shortest = strs[i];
			}
		}
		String res = shortest;
		for (int i = 0;i < strs.length;i++) {
			String temp = "";
			for (int j = 0;j < res.length();j++) {
				if (res.charAt(j) == strs[i].charAt(j)) {
					temp += res.charAt(j);
				} else
					break;
			}
			res = temp;
		}
		
		return res;
	}
	
	/**
	 * 先排序，然后左右各一个指针移动
	 * @param nums
	 * @return
	 */
	public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3)
            return res;
        Arrays.sort(nums);
        
        for (int i = 0;i < nums.length-2;i++) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            
            while (left < right) {
                if (nums[i] + nums[left] + nums[right] > 0) {
                    right--;
                    while (nums[right] == nums[right + 1] && left < right) right--;
                } else if (nums[i] + nums[left] + nums[right] < 0) {
                    left++;
                    while (nums[left] == nums[left - 1] && left < right) left++;
                } else {
                    res.add(Arrays.asList(nums[left], nums[right], nums[i]));
                    ++left;
                    --right;
                    while (nums[right] == nums[right+1] && left < right) right--;
                    while (nums[left] == nums[left-1] && left < right) left++;
                }
            }
            
        }
        
        return res;
    }
	
	/**
	 * 三数之和与目标值最近
	 * @param nums
	 * @param target
	 * @return
	 */
	public int threeSumClosest(int[] nums, int target) {
        if (nums.length < 3) return 0;
        Arrays.sort(nums);
        int interval = Integer.MAX_VALUE;
        int result = 0;
        // 遍历目标值，定义left和right两个指针
        // 左右夹逼
        for (int i = 0;i < nums.length;i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            
            int left = i + 1;
            int right = nums.length - 1;
            
            while (left < right) {
            	// 计算和与目标值之间的距离
                int temp = nums[i] + nums[left] + nums[right];
                if (temp == target) {
                    result = temp;
                    break;
                }
                int gap =Math.abs(temp - target);
                
                if (gap < interval) {
                    result = temp;
                    interval = gap;
                } 
                if (temp > target) {
                    right--;
                    while (nums[right] == nums[right + 1] && left < right) right--;
                } else {
                    left++;
                    while (nums[left] == nums[left - 1] && left < right) left++;
                }
            }
        }
        return result;
    }
	
	/**
	 * 双队列结构
	 * 1. 终止条件，当队首节点值与字符串的值不相等的时候
	 * 2. 移除队首节点，初始化为对应数字代表的字符串，依次插入队尾，如‘a’->'b'->'c'
	 * 3. 取出队首节点，与后一位数字代表的字符串依次连接，如'ad'->'ae'->'af',插入到队尾
	 * 4. 重复该过程，知道队首字符串长度与digits相等
	 * @param digits
	 * @return
	 */
	public List<String> letterCombination(String digits) {
		LinkedList<String> ans = new LinkedList<String>();
		if(digits.isEmpty()) return ans;
		String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
		ans.add("");
		while(ans.peek().length()!=digits.length()){
			String remove = ans.remove();
			String map = mapping[digits.charAt(remove.length())-'0'];
			for(char c: map.toCharArray()){
				ans.addLast(remove+c);
			}
		}
		return ans;
	}
	
	public List<List<Integer>> fourSum(int[] nums, int target) {
		ArrayList<List<Integer>> res = new ArrayList<>();
		int len = nums.length;
		if (nums == null || len < 4) {
			return res;
		}
		
		Arrays.sort(nums);
		int max = nums[len - 1];
		if (4 * nums[0] > target || 4 * max < target) 
			return res;
		int i, z;
		for (i = 0;i < len;i++) {
			z = nums[i];
			if (i > 0 && z == nums[i - 1])
				continue;
			if (z + 3 * max < target) //z太小
				continue;
			if (4 * z > target)		// z太大
				break;
			if (4 * z == target) {
				if (i + 3 < len && nums[i+3] == z)
					res.add(Arrays.asList(z,z,z,z));
				break;
			}
			
			for (int j = i + 1;j < len-1;j++) {
				if (j > 0 && nums[j] == nums[j - 1])
					continue;
				int left = j + 1;
				int right = len - 2;
				while (left < right) {
					int sum = nums[i] + nums[j] + nums[left] + nums[right];
					if (sum < target) {
						left++;
						while (nums[left] == nums[left-1]) left++;
					} else if (sum > target) {
						right--;
						while (nums[right] == nums[right+1]) right--;
					} else {
						res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
						// 跳过重复值，不加会超时
						++left;
	                    --right;
	                    while (nums[right] == nums[right+1] && left < right) right--;
	                    while (nums[left] == nums[left-1] && left < right) left++;
					}
				}
			}
			
		}
		return res;
	}

	
	
	public ListNode deleteX(ListNode head, int x) {
		List<List<Integer>> res = new ArrayList<>();
		if (head == null) return null;
		// 当一开始有一些x的节点
		
		ListNode q;
		if (head.val == x) {
			head = head.next;
			return deleteX(head, x);
		} else {
			head.next = deleteX(head.next, x);
		}
		
		return head;
	}
	
	/**
	 * 拓展: 最长的不重复子串
	 * 在求出最长子串长度的基础上，更新起始位置
	 * @param s
	 */
	public void LongestSub(String s) {
		HashMap<Character, Integer> map = new HashMap<>();
		char[] arr = s.toCharArray();
		int maxLength = 0;
		int originalIndex = -1;
		for (int startIndex = 0, i = 0;i < arr.length;i++) {
			if (map.containsKey(arr[i])) {
				startIndex = Math.max(map.get(arr[i]), startIndex);
			}
			if (i - startIndex + 1 > maxLength) {
				maxLength = i - startIndex + 1;
				// 更新起始位置的索引
				originalIndex = startIndex;
			}
			
			map.put(arr[i], i + 1);
		}
		System.out.println(maxLength);
		for (int i = originalIndex;i < maxLength + originalIndex;i++) {
			System.out.print(arr[i]);
		}
	} 
	
	/**
	 * 两个有序数组的中位数
	 * 数组A： A[0],A[1]...A[i-1]  | A[i], A[i+1],...,A[n-1]
	 * 数组B：B[0],B[1]...B[j-1] | B[j], B[j+1],...,B[n-1]
	 * 把A B数组都分成两部分，如果我们能保证左右两部分长度相等，而且左边的最大值小于等于右边的最小值，那么median就是
	 * max(left_part) + min(right_part) / 2
	 * 为了保证左右两部分长度相等， 则  i+j = m - i + n - j
	 * 我们可以遍历 i: 0->m,  j = (m + n) / 2 - i
	 * 为了保证左边的最大值小于等于右边的最小值，则
	 * B[j-1] <= A[i]  且  A[i-1] <= B[j]
	 * @param A
	 * @param B
	 * @return
	 */
	public double findMedianNum(int[] A, int[] B) {
		int m = A.length;
		int n = B.length;
		// 确保短数组为A
		if (m > n) {
			int[] temp = A; A =B;B=temp;
			int tempL = m;m=n;n=tempL;
		}
		int start = 0;
		int end = m;
		int middle = (m+n+1)/2;
		while (start <= end) {
			int i = (start + end) / 2;
			int j = middle - i;
			// 当B[j-1]>A[i]时，说明i太小了，start++;
			if (i > 0 && B[j - 1] > A[i]) {
				start = start + 1;
			}	// 这里说明i太大了， end--;
			else if (i < end && A[i - 1] > B[j]) {
				end = end - 1;
			} else {
				int maxLeft = 0;
				// i为0，A数组中所有元素都大于B
				// j为0， B[j]前面的数都小于A[i-1]
				if (i == 0) maxLeft = B[j - 1];
				else if (j == 0) maxLeft = A[i - 1];
				// 一般情况的时候，取更大的值
				else { maxLeft = Math.max(A[i-1], B[j-1]);}
				if ((m + n) % 2 == 1) {return maxLeft;}
				int minRight = 0;
				if (i == m) {
					minRight = B[j];
				} else if (j == n) {
					minRight = A[i];
				} else {
					minRight = Math.min(A[i], B[j]);
				}
				return (minRight + maxLeft) / 2.0;
			}
			
		}
		return 0.0;
	}
	
	/**
	 * 删除倒数第n个节点
	 * @param head
	 * @param n
	 * @return
	 */
	public ListNode removeNthFromEnd(ListNode head, int n) {
		if (head == null)
			return null;
		ListNode start = head;
		ListNode deleted = head;
        ListNode prev = null;
		for (int i = 0;i < n;i++) {
			if (start == null) break;
			start = start.next;
			// else {
			// 	if (i < n - 1)
			// 		return head;
			// 	break;
			// }
		}
		while (start != null) {
            prev = deleted;
			deleted = deleted.next;
			start = start.next;
		}
		// 头结点
		if (deleted == head) {
			head = head.next;
		} else if (deleted.next == null) {
			// 尾节点
            prev.next = null;
			deleted = null;
		} else {
			// 中间节点
			ListNode p = deleted.next;
			deleted.val = p.val;
			deleted.next = p.next;
			p = null;
		}
		return head;
	}
	
	/**
	 * 回溯法生成括号
	 * 用两个变量记录已使用的左括号和右括号的数目
	 * 字符串结果每使用一次右括号，右括号数目减1
	 * 没使用一次左括号，左括号数目减1，右括号数目加1
	 * 二者均为0时，添加到结果中，return
	 * @author Administrator
	 *
	 */
	public List<String> generateParenthesis(int n) {
		List<String> res = new LinkedList<>();
		addPar(res, "", n, 0);
		return res;
	}
	
	public void addPar(List<String> res, String str, int n, int m) {
		if (n == 0 && m == 0) {
			res.add(str);
			return;
		}
		if (m > 0) {
			addPar(res, str + ')', n, m - 1);
		}
		if (n > 0) {
			addPar(res, str + '(', n - 1, m + 1);
		}
	}
	
	/**
	 * 
	 * @param lists
	 * @return
	 */
	public ListNode mergeKLists(ListNode[] lists) {
		if (lists.length == 0) {
			return null;
		}
		return merge(0, lists.length - 1, lists);
	}
	
	public ListNode merge(int start, int end, ListNode[] lists) {
		if (end < start) return null;
		if (start == end) return lists[start];
		int mid = (start + end) / 2;
		ListNode l = merge(start, mid, lists);
		ListNode r = merge(mid + 1, end, lists);
		ListNode dummy = new ListNode(0);
		ListNode runner = dummy;
		
		while (l != null && r != null) {
			if (l.val > r.val) {
				runner.next = r;
				r = r.next;
				runner = runner.next;
			} else {
				runner.next = l;
				l = l.next;
				runner = runner.next;
			}
		}
		if (l == null && r==null) {
			return dummy.next;
		}
		runner.next = l == null?r : l;
		return dummy.next;
	}
	/**
	 * 
	 * @param head
	 * @return
	 */
	public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(-1);
        
        int n =1;
        ListNode prev = head;
        ListNode cur = head;
        dummy.next = cur;
        ListNode res = dummy;
        while (cur != null) {
            if (n != 2) {
                prev = cur;
                cur = cur.next;
                n++;
            } else {
                ListNode d = reverse(prev, cur);
                res.next = cur;
                cur = prev.next;
                res = prev;
                n = 1;
            }
        }
        
        return dummy.next;
    }
    
    public ListNode reverse(ListNode l1, ListNode l2) {
        if (l1 != null && l2 != null) {
            ListNode temp = l2.next;
            l2.next = l1;
            l1.next = temp;
            l1 = l2;
            l2 = temp;
        }
        return l1;
    }
	
	/*
	 * 递归版本的
	 * 
	 */
	public ListNode reverseKGroups(ListNode head, int k) {
//		ListNode curr = head;
//		int count = 0;
//		// 找到第k+1个节点
//		while (count != k && curr != null) {
//			count++;
//			curr = curr.next;
//		}
//		// 第k个节点
//		if (count == k) {
//			// 翻转第k组
//			ListNode t = reverseKGroups(curr, k);
//			// 翻转的代码
//			//  a->b->c->d
//			while (count > 0) {
//				ListNode temp = head.next;	
//				head.next = t;	// a->d, b->a, c->b最后  c->b->a->d
//				t = head;		
//				head = temp;	
//				count--;
//			}
//			head = t;
//		}
//		
//		return head;
		
		// 非递归形式
		/*
		 * Dummy -> 1 -> 2 -> 3 -> 4 -> 5
   			  p     c         n
         		  start
		   Dummy -> 2 -> 3 -> 1 -> 4 -> 5
   			  p     c    n    start
		   Dummy -> 3 -> 2 -> 1 -> 4 -> 5
   			  p     c         start
         			n
		 */
		ListNode dummy = new ListNode(0), start = dummy;
        dummy.next = head;
        while(true) {
            ListNode p = start, c, n = p;
            start = p.next;
            for(int i = 0; i < k && n != null; i++) n = n.next;
            if(n == null) break;
            for(int i = 0; i < k-1; i++) {
                c = p.next;
                p.next = c.next;
                c.next = n.next;
                n.next = c;
            }
        }
        return dummy.next;
	}
	
	/**
	 * l中字符长度都一样
	 * @param s
	 * @param l
	 * @return
	 */
	public List<Integer> findSubstring(String s, String[] l) {
		List<Integer> res = new ArrayList<>();
		if (s == null || l == null || l.length == 0) return res;
		int len = l[0].length();
		Map<String, Integer> map = new HashMap<>();
		
		for (String w:l) {
			map.put(w, map.containsKey(w) ? map.get(w) +1:1);
		}
		
		for (int i = 0;i <= s.length() - len * l.length;i++) {
			Map<String, Integer> copy = new HashMap<>(map);
			for (int j = 0;j < l.length;j++) {
				String str = s.substring(i + j * len, i + j * len + len);
				if (copy.containsKey(str)) {
					int count = copy.get(str);
					if (count == 1) copy.remove(str);
					else copy.put(str, count-1);
					if (copy.isEmpty()) {
						res.add(i);
						break;
					}
				} else break;
			}
		}
		return res;
	}
	
	
	private static class ListNode {
    	int val;
    	ListNode next = null;
    	public ListNode(int val) {
    		this.val = val;
		}
    }
    
    
    class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }

    private static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
        TreeNode next = null;
        
        TreeNode(int val) {
            this.val = val;
        }
    }

}
