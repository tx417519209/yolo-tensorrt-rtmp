// 头文件防卫
#ifndef CIRCULAR_QUEUE_H
#define CIRCULAR_QUEUE_H

#include <mutex> // 互斥量
#include <condition_variable> // 条件变量

template <typename T>
class CircularQueue {
public:
    // 构造函数，初始化成员变量
    explicit CircularQueue(size_t capacity) :
        capacity_(capacity),
        size_(0),
        head_(0),
        tail_(0),
        buffer_(new T[capacity]) {}

    // 析构函数，释放 buffer_ 内存
    ~CircularQueue() {
        not_full_.notify_all();
        not_empty_.notify_all();
        delete[] buffer_;
    }

    // 判断队列是否为空
    bool empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    // 判断队列是否已满
    bool full() {
        std::unique_lock<std::mutex> lock(mutex_);
        return size_ == capacity_;
    }

    // 获取队列中元素的数量
    size_t size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return size_;
    }

    // 获取队列的容量
    size_t capacity() {
        return capacity_;
    }

    // 将元素加入队列，可能会阻塞
    bool push(const T& value, bool block = true) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (block) {
            // 如果队列已满，则等待队列不满
          not_full_.wait(lock, [=]() {
            return size_ < capacity_;
           });
        } else {
            // 如果队列已满，则返回 false
            if (size_ == capacity_) {
                return false;
            }
        }

        // 将元素加入队列尾部，并更新 tail_ 和 size_
        buffer_[tail_] = value;
        tail_ = (tail_ + 1) % capacity_;
        ++size_;

        // 通知一个等待在 not_empty_ 条件变量上的线程
        not_empty_.notify_one();

        return true;
    }

    // 将元素加入队列，可能会阻塞，使用右值引用
    bool push(T&& value, bool block = true) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (block) {
            // 如果队列已满，则等待队列不满
            while (size_ == capacity_) {
                not_full_.wait(lock);
            }
        } else {
            // 如果队列已满，则返回 false
            if (size_ == capacity_) {
                return false;
            }
        }

        // 将元素加入队列尾部，并更新 tail_ 和 size_
        buffer_[tail_] = std::move(value);
        tail_ = (tail_ + 1) % capacity_;
        ++size_;

        // 通知一个等待在 not_empty_ 条件变量上的线程
        not_empty_.notify_one();

        return true;
    }

    // 从队列中取出元素，可能会阻塞
    bool pop(T& value, bool block = true) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (block) {
            // 如果队列为空，则等待队列不空
          not_empty_.wait(lock, [=]() {
            return size_ > 0;
          });
         
           
        } else {
            // 如果队列为空，则返回 false
            if (size_ == 0) {
                return false;
            }
        }

        // 取出队列头部元素，并更新 head_ 和 size_
        value = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        --size_;

        // 通知一个等待在 not_full_ 条件变量上的线程
        not_full_.notify_one();

        return true;
    }

    T pop() {
      std::unique_lock<std::mutex> lock(mutex_);
      auto status=not_empty_.wait_for(lock,std::chrono::milliseconds(50),[=]() {
          return size_ > 0;
       });

      if(!status){
          return cv::Mat();
      }
      auto value = buffer_[head_];
      head_ = (head_ + 1) % capacity_;
      --size_;

      not_full_.notify_one();

      return value;
    }


private:
    const size_t capacity_; // 队列容量
    size_t size_; // 队列中元素的数量
    size_t head_; // 队列头部指针
    size_t tail_; // 队列尾部指针
    T* buffer_; // 队列缓冲区    
    std::mutex mutex_; // 互斥量，保护队列缓冲区和队列大小
    std::condition_variable not_full_; // 条件变量，当队列满时等待
    std::condition_variable not_empty_; // 条件变量，当队列空时等待
};

#endif // CIRCULAR_QUEUE_H