#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XHand ZMQ Forwarder
Runs in the XHand environment and forwards hand control/observation data via ZMQ
"""

import argparse
import sys, os
import time
import yaml
import logging
import json

import numpy as np
import zmq

# XHand-specific imports will be added at runtime
# from xhand_tele_ops import XHandTeleOps


# Joint limits (radians) per joint index 0..11 based on the provided spec image
# Mapping assumption:
# 0: thumb_bend_joint           [0, 105°]
# 1: thumb_rota_joint1          [-40°, 90°]
# 2: thumb_rota_joint2          [0, 90°]
# 3: index_bend_joint           [-10°, 10°]
# 4: index_joint1               [0, 110°]
# 5: index_joint2               [0, 110°]
# 6: mid_joint1                 [0, 110°]
# 7: mid_joint2                 [0, 110°]
# 8: ring_joint1                [0, 110°]
# 9: ring_joint2                [0, 110°]
# 10: pinky_joint1              [0, 110°]
# 11: pinky_joint2              [0, 110°]
JOINT_LIMITS_RAD = [
    (0.0, 1.832595715),     # 105°
    (-0.698131701, 1.570796327),  # -40° ~ 90°
    (0.0, 1.570796327),     # 90°
    (-0.174532925, 0.174532925),  # -10° ~ 10°
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
]


def _clamp_hand_action(hand_action):
    """Clamp a 12-DoF hand action (radians) to JOINT_LIMITS_RAD.
    Pads/truncates to 12 dims if needed and logs a warning.
    Returns a list of 12 floats.
    """
    arr = np.asarray(hand_action, dtype=np.float32)
    mins = np.array([mn for mn, _ in JOINT_LIMITS_RAD], dtype=np.float32)
    maxs = np.array([mx for _, mx in JOINT_LIMITS_RAD], dtype=np.float32)
    clipped = np.clip(arr, mins, maxs)
    return clipped.tolist()


class XHandForwarder:
    """ZMQ forwarder for XHand communication"""
    
    def __init__(self, config_path, zmq_port=5557):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        logging.info(f"XHand forwarder listening on port {zmq_port}")
        
        # XHand configuration
        self.xhand_code_path = self.config['xhand_code_path']
        self.xhand_config = self.config['xhand_config']
        
        # Initialize XHand controller
        self._initialize_xhand()
    
    def _initialize_xhand(self):
        """Initialize the XHand controller"""
        os.chdir(self.xhand_code_path)
        from xhand_tele_ops import XHandTeleOps
        
        self.xhand_controller = XHandTeleOps(self.xhand_config)
        logging.info("XHand controller initialized successfully")
    
    def get_observations(self):
        """Get current hand state observations"""
        observations = {}
        
        # Get left hand data
        resp = self.xhand_controller.get_hand_full_info("hand_a", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            left_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
            observations['left_hand'] = left_hand_pos
        else:
            logging.error("Failed to get left hand data")
            observations['left_hand'] = [0.0] * 12
        
        # Get right hand data
        resp = self.xhand_controller.get_hand_full_info("hand_b", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            right_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
            observations['right_hand'] = right_hand_pos
        else:
            logging.error("Failed to get right hand data")
            observations['right_hand'] = [0.0] * 12
        
        return observations
    
    def execute_action(self, action_data):
        """Execute hand actions"""
        try:
            # Extract hand actions
            left_hand_action = action_data.get('left_hand', [0.0] * 12)
            right_hand_action = action_data.get('right_hand', [0.0] * 12)
            
            # Clamp actions to safe joint limits (radians)
            left_hand_action = _clamp_hand_action(left_hand_action)
            right_hand_action = _clamp_hand_action(right_hand_action)

            # Prepare data for xhand controller
            transform_data = {
                "left_hand": left_hand_action,
                "right_hand": right_hand_action
            }
            
            # Send actions to xhand
            logging.debug(f"Executing xHand action: L={left_hand_action}, R={right_hand_action}")
            self.xhand_controller.send_data_xhand(transform_data)
            
            return {'status': 'success'}
            
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            return {'error': str(e)}
    
    def handle_request(self, request):
        """Handle incoming ZMQ request"""
        command = request.get('command')
        
        if command == 'get_observations':
            return self.get_observations()
        
        elif command == 'execute_action':
            action_data = request.get('action_data')
            return self.execute_action(action_data)
        
        elif command == 'ping':
            return {'status': 'pong'}
        
        else:
            return {'error': f'Unknown command: {command}'}
    
    def run(self):
        """Main loop for handling requests"""
        logging.info("XHand forwarder started")
        
        try:
            while True:
                # Wait for request
                message = self.socket.recv_json()
                logging.debug(f"Received request: {message.get('command')}")
                
                # Process request
                try:
                    response = self.handle_request(message)
                except Exception as e:
                    logging.error(f"Error handling request: {e}")
                    response = {'error': str(e)}
                
                # Send response
                self.socket.send_json(response)
                
        except KeyboardInterrupt:
            logging.info("Shutting down XHand forwarder...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        # Close ZMQ socket
        self.socket.close()
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description="XHand ZMQ Forwarder")
    parser.add_argument("--config", type=str, default="deploy/mmk_xhand_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--port", type=int, default=5557,
                        help="ZMQ port to listen on")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run forwarder
    forwarder = XHandForwarder(args.config, args.port)
    forwarder.run()


if __name__ == "__main__":
    main()
