import java.util.HashMap;

public class LeetCode {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
//		ListNode node1 = new ListNode(1);
//	
//		ListNode node2 = new ListNode(9);
//		node2.next = new ListNode(9);
		LeetCode leetCode = new LeetCode();
//		leetCode.addTwoNumbers(node1, node2);
		leetCode.LongestSub("abcdab");
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
	 * 拓展: 最长的不重复子串
	 * 在求出最长子串长度的基础上，更新起始位置
	 * @param s
	 */
	public void LongestSub(String s) {
//		int i, j;
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
